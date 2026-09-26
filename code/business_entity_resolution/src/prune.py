"""Stage-1 candidate pruning: a cheap model keeps only each S2/S3 record's few best S1 candidates.

The final decision picks one S1 per S2/S3 record, so candidates far below a record's best
are almost never matches. A small model on the blocking/embedding scores only (no string
features) ranks each record's candidates; a pair is kept if it is among the record's top K and
its stage-1 probability is >= P_MIN. The kept set is what features.py / the final model see,
i.e. exactly what candidate_pairs.tsv reports. p1 and p1rank are kept as stage-2 features.

train: two stage-1 models on alternating candidate parts, each scoring the other half
       (out-of-fold). K / P_MIN are chosen on the training labels: the smallest candidate set
       that keeps recall within MAX_LOSS of the unpruned set. Saved to WORK/prune.json.
test:  the saved stage-1 model and K / P_MIN are applied.

Usage: python prune.py train|test [--sample FRACTION]
Input/output: WORK/{split}_cand[_sample].parquet (the unpruned set is kept as
..._union.parquet so a rerun starts from it).
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import polars as pl

import model
from config import WORK
from features import context

P1_FEATURES = ["bscore", "n_shared", "brank", "bscore_rel", "bscore_gap",
               "emb_sim", "emb_gap", "erank"]
KS = (1, 2, 3, 4, 5)
P_MINS = (0.0, 0.001, 0.003, 0.01, 0.03)
MAX_LOSS = 0.002       # allowed drop in candidate recall of true pairs vs unpruned
OVERRIDE = os.environ.get("BER_PRUNE")   # "K,p_min" to force a setting (train only)
ROUNDS = 300
TRAIN_ROWS = 20_000_000   # stage-1 training rows per half (cheap model, plenty of data)


def labelled(cand: pl.DataFrame) -> pl.DataFrame:
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "entity_id"])
    s23 = pl.read_parquet(WORK / "train_s23.parquet", columns=["k", "entity_id"])
    t = (pl.read_parquet(WORK / "train_truth.parquet")
         .join(s23.select(s23k="k", s23_id="entity_id"), on="s23_id")
         .join(s1.select(s1k="k", s1_id="entity_id"), on="s1_id")
         .select("s23k", "s1k", y=pl.lit(1, pl.Int8)))
    return cand.join(t, on=["s23k", "s1k"], how="left").with_columns(pl.col("y").fill_null(0))


def X(df):
    return df.select(pl.col(P1_FEATURES).cast(pl.Float32)).to_numpy()


def rank(df: pl.DataFrame, p) -> pl.DataFrame:
    return (df.with_columns(p1=pl.Series(p, dtype=pl.Float32))
            .with_columns(p1rank=pl.col("p1").rank("ordinal", descending=True)
                          .over("s23k").cast(pl.UInt8)))


def run(split, sample=None):
    t0 = time.time()
    name = f"{split}_cand{'_sample' if sample else ''}"
    union = WORK / f"{name}_union.parquet"
    if "p1" not in pl.read_parquet_schema(WORK / f"{name}.parquet"):
        shutil.copy(WORK / f"{name}.parquet", union)   # fresh embed.py output
    cand = context(pl.read_parquet(union))
    assert "emb_sim" in cand.columns, "run embed.py before prune.py"
    print(f"{name}: {len(cand):,} pairs, {cand['s23k'].n_unique():,} records", flush=True)

    if split == "train":
        cand = labelled(cand)
        half = (pl.col("s23k").hash(11) % 2).cast(pl.Int8)
        cand = cand.with_columns(half=half)
        p = np.zeros(len(cand), dtype=np.float32)
        for h in (0, 1):
            tr = cand.filter(pl.col("half") == h)
            if len(tr) > TRAIN_ROWS:
                tr = tr.sample(TRAIN_ROWS, seed=h)
            m, _ = model.fit(X(tr), tr["y"].to_numpy(), P1_FEATURES, rounds=ROUNDS)
            idx = np.flatnonzero((cand["half"] != h).to_numpy())
            p[idx] = model.predict(m, X(cand[idx]))
            if h == 0:
                model.save(m, WORK, "prune")      # used for test
            print(f"  stage-1 half {h} {time.time() - t0:.0f}s", flush=True)
        cand = rank(cand, p)
        n_true = int(cand["y"].sum())
        n_rec = cand["s23k"].n_unique()
        rows, best = [], None
        for k in KS:
            for pm in P_MINS:
                kept = cand.filter((pl.col("p1rank") <= k) & (pl.col("p1") >= pm))
                rec = kept["y"].sum() / n_true
                rows.append((k, pm, round(rec, 5), round(len(kept) / n_rec, 2), len(kept)))
        print(pl.DataFrame(rows, schema=["K", "p_min", "recall_kept", "pairs_per_record", "pairs"],
                           orient="row"))
        ok = [r for r in rows if r[2] >= 1 - MAX_LOSS]
        k, pm = min(ok, key=lambda r: r[4])[:2] if ok else (max(KS), 0.0)
        if OVERRIDE:
            k, pm = int(OVERRIDE.split(",")[0]), float(OVERRIDE.split(",")[1])
        (WORK / "prune.json").write_text(json.dumps({"K": k, "p_min": pm, "model": model.KIND}))
        cand = cand.drop("y", "half")
    else:
        cfg = json.loads((WORK / "prune.json").read_text())
        k, pm = cfg["K"], cfg["p_min"]
        m = model.load(WORK, cfg["model"], "prune")
        p = np.concatenate([model.predict(m, X(cand.slice(i, 5_000_000)), kind=cfg["model"])
                            for i in range(0, len(cand), 5_000_000)])
        cand = rank(cand, p)

    kept = (cand.filter((pl.col("p1rank") <= k) & (pl.col("p1") >= pm))
            .drop("bscore_rel", "bscore_gap"))
    kept.write_parquet(WORK / f"{name}.parquet")
    n_s1 = kept["s1k"].n_unique()
    print(f"kept K={k} p_min={pm}: {len(kept):,} of {len(cand):,} pairs "
          f"({len(kept) / cand['s23k'].n_unique():.2f} per S2/S3 record, "
          f"{len(kept) / max(n_s1, 1):.2f} per S1 with candidates) {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--sample", type=float)
    a = ap.parse_args()
    run(a.split, a.sample)
