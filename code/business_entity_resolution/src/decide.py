"""Decision tuning on out-of-fold scores, then re-apply to the test scores (no retraining).

The base rule (train.py) assigns each S2/S3 record to its best S1 if p > t. Under F0.5 a wrong
merge costs ~3x a missed link, so three precision guards are tuned on the train OOF scores
(coordinate ascent on the official macro F0.5 over all train S1):
  t        global threshold on the record's best probability
  margin   the best S1 must beat the record's second-best S1 by at least this much (near-ties
           between two businesses are where decoys win)
  s1_min   an S1 whose strongest assigned record is below s1_min gets no matches at all
           (protects singletons, which score 0 on any false match)
  t_<country>  per-country threshold override (countries seen in training); other countries
           (France) use the global t
Writes WORK/decision_tuned.json and, from WORK/test_scored.parquet, OUTPUT/matching_results.tsv
(candidate_pairs.tsv is unchanged: the candidate set is the same).

Usage: python decide.py [train_sample_fraction]   (1.0 = full train, the default)
"""
import json
import sys
import time

import numpy as np
import polars as pl

from config import OUTPUT, WORK
from metric import macro_f05
from train import truth_pairs

T_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
MARGINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
S1_MINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def top2(scored: pl.DataFrame, s1_country: pl.DataFrame) -> pl.DataFrame:
    """One row per S2/S3 record: best S1, its p, the runner-up p, and the S1's country."""
    s = scored.sort(["s23k", "p"], descending=[False, True])
    return (s.group_by("s23k", maintain_order=True)
            .agg(pl.col("s1k").first(), p1=pl.col("p").first(),
                 p2=pl.col("p").slice(1, 1).first().fill_null(0.0))
            .join(s1_country, left_on="s1k", right_on="k", how="left"))


def apply(tops: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    t = pl.lit(cfg["t"])
    for c, v in cfg.get("t_country", {}).items():
        t = pl.when(pl.col("country") == c).then(pl.lit(v)).otherwise(t)
    kept = tops.filter((pl.col("p1") > t) & (pl.col("p1") - pl.col("p2") >= cfg["margin"]))
    if cfg["s1_min"] > 0:
        strong = kept.group_by("s1k").agg(pl.col("p1").max().alias("pmax"))
        kept = kept.join(strong.filter(pl.col("pmax") >= cfg["s1_min"]).select("s1k"), on="s1k")
    return kept.select("s23k", "s1k")


def main(fraction=1.0):
    t0 = time.time()
    truth = truth_pairs()
    oof = pl.read_parquet(WORK / "oof.parquet").select("s23k", "s1k", "p")
    s1c = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "country"])
    from block import sample_s1_keys
    keys = sample_s1_keys(fraction)
    tops = top2(oof, s1c)
    print(f"{len(tops):,} records with candidates  {time.time() - t0:.0f}s", flush=True)

    def score(cfg):
        return macro_f05(truth, apply(tops, cfg), keys)["macro_f05"]

    base = json.loads((WORK / "decision.json").read_text())
    cfg = {"t": base["threshold"], "margin": 0.0, "s1_min": 0.0, "t_country": {}}
    best = score(cfg)
    print(f"base rule t={cfg['t']}: {best}")
    for _ in range(2):                                  # two rounds of coordinate ascent
        for name, grid in (("t", T_GRID), ("margin", MARGINS), ("s1_min", S1_MINS)):
            for v in grid:
                trial = {**cfg, name: v}
                r = score(trial)
                if r > best:
                    best, cfg = r, trial
            print(f"  {name} -> {cfg[name]}  macro_f05={best}", flush=True)
        for c in s1c["country"].unique().sort().to_list():
            for v in T_GRID:
                trial = {**cfg, "t_country": {**cfg["t_country"], c: v}}
                r = score(trial)
                if r > best:
                    best, cfg = r, trial
            print(f"  t_{c} -> {cfg['t_country'].get(c, cfg['t'])}  macro_f05={best}", flush=True)

    final = apply(tops, cfg)
    print("tuned:", cfg)
    print("overall:", macro_f05(truth, final, keys))
    for c in s1c["country"].unique().sort().to_list():
        ck = s1c.filter((pl.col("country") == c) & pl.col("k").is_in(keys))["k"]
        print(f"  {c}: {macro_f05(truth, final, ck)}")
    (WORK / "decision_tuned.json").write_text(json.dumps({**cfg, "cv": best, "cv_base": base["cv"]}))

    # re-apply to test
    test = pl.read_parquet(WORK / "test_scored.parquet").select("s23k", "s1k", "p")
    s1t = pl.read_parquet(WORK / "test_s1.parquet", columns=["k", "entity_id", "country"]).sort("k")
    matches = apply(top2(test, s1t.select("k", "country")), cfg)
    base_n = len(top2(test, s1t.select("k", "country")).filter(pl.col("p1") > base["threshold"]))
    s23 = pl.read_parquet(WORK / "test_s23.parquet", columns=["k", "entity_id"])
    from predict import write_lists
    write_lists(matches.lazy(), s1t.select("k", "entity_id"), s23, "matched_entity_ids",
                OUTPUT / "matching_results.tsv")
    per = matches.join(s1t.select("k", "country"), left_on="s1k", right_on="k")
    print(f"test matches {len(matches):,} (base rule {base_n:,}); per country:",
          per.group_by("country").len().sort("country").rows())
    print(f"wrote {OUTPUT / 'matching_results.tsv'}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
