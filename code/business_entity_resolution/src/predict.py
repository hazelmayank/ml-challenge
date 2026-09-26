"""Score test candidates and write the two submission files (streaming, low memory).

Feature parts are grouped by S2/S3 key ranges, so the per-record decision (argmax over a
record's candidates, then threshold) can be made part by part.
Usage: python predict.py
Outputs: OUTPUT/matching_results.tsv, OUTPUT/candidate_pairs.tsv, WORK/test_scored.parquet
"""
import json
import time

import numpy as np
import polars as pl

import model as backend
from config import OUTPUT, WORK
from features import RECORDS
from train import decide

S1_BATCH = 200_000


def write_lists(pairs: pl.LazyFrame, s1: pl.DataFrame, s23: pl.DataFrame, col: str, path):
    """One row per S1 entity (all of them, in S1 order), comma-joined S2/S3 ids."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for lo in range(0, len(s1), S1_BATCH):
            batch = s1.slice(lo, S1_BATCH)
            lists = (pairs.filter(pl.col("s1k").is_between(batch["k"].min(), batch["k"].max()))
                     .collect()
                     .join(s23, left_on="s23k", right_on="k")
                     .group_by("s1k")
                     .agg(pl.col("entity_id").unique().sort().str.join(",").alias(col)))
            out = (batch.join(lists, left_on="k", right_on="s1k", how="left")
                   .select("entity_id", pl.col(col).fill_null("")))
            f.write(out.write_csv(separator="\t", include_header=False, quote_style="never"))


def main():
    t0 = time.time()
    decision = json.loads((WORK / "decision.json").read_text())
    t, kind = decision["threshold"], decision.get("model", "lgb")
    features = decision.get("features") or __import__("features").FEATURES
    model = backend.load(WORK, kind)
    print(f"model {kind}, {len(features)} features, t={t}", flush=True)
    parts = sorted((WORK / "test_cand_feat_parts").glob(f"*_{RECORDS}.parquet"))
    scored_parts, match_parts = [], []
    for i, part in enumerate(parts):
        feat = pl.read_parquet(part)
        p = backend.predict(model, feat.select(pl.col(features).cast(pl.Float32)).to_numpy(), kind=kind)
        scored = feat.select("s23k", "s1k").with_columns(p=pl.Series(p, dtype=pl.Float32))
        scored_parts.append(scored)
        match_parts.append(decide(scored, t))
        if i % 10 == 0:
            print(f"  part {i + 1}/{len(parts)} {time.time() - t0:.0f}s", flush=True)
    scored = pl.concat(scored_parts)
    matches = pl.concat(match_parts)
    del scored_parts, match_parts
    scored.write_parquet(WORK / "test_scored.parquet")
    print(f"scored {len(scored):,} pairs, {len(matches):,} matches (t={t}) "
          f"{time.time() - t0:.0f}s", flush=True)
    del scored

    s1 = pl.read_parquet(WORK / "test_s1.parquet", columns=["k", "entity_id"]).sort("k")
    s23 = pl.read_parquet(WORK / "test_s23.parquet", columns=["k", "entity_id"])
    write_lists(matches.lazy(), s1, s23, "matched_entity_ids",
                OUTPUT / "matching_results.tsv")
    write_lists(pl.scan_parquet(WORK / "test_scored.parquet").select("s23k", "s1k"),
                s1, s23, "candidate_entity_ids", OUTPUT / "candidate_pairs.tsv")
    print(f"wrote {OUTPUT} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
