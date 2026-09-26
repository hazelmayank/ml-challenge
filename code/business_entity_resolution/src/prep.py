"""Read raw TSVs, normalise them, and store as parquet in WORK.

Outputs per split (train/test):
  {split}_s1.parquet   Source 1 records (+ integer row key `k`)
  {split}_s23.parquet  Source 2 + 3 records (+ `k`, `src`)
  train_truth.parquet  s23 entity_id -> true s1 entity_id (only linked records)
Usage: python prep.py [train|test ...]
"""
import sys
import time

import polars as pl

from config import DATA, WORK
from normalize import normalize

READ = dict(separator="\t", quote_char=None, infer_schema_length=0, encoding="utf8",
            missing_utf8_is_empty_string=True)


def read_source(split, n):
    df = pl.read_csv(DATA / split / f"{split}_source{n}.tsv", **READ)
    return normalize(df).with_columns(src=pl.lit(n, pl.Int8))


def prep_split(split):
    t = time.time()
    s1 = read_source(split, 1).with_row_index("k")
    s1.write_parquet(WORK / f"{split}_s1.parquet")
    print(f"{split} s1 {s1.shape} {time.time() - t:.0f}s", flush=True)
    del s1
    s23 = pl.concat([read_source(split, 2), read_source(split, 3)]).with_row_index("k")
    s23.write_parquet(WORK / f"{split}_s23.parquet")
    print(f"{split} s23 {s23.shape} {time.time() - t:.0f}s", flush=True)


def prep_truth():
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", **READ)
    truth = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
             .explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "")
             .select(s23_id="matched_entity_ids", s1_id="source1_entity_id"))
    truth.write_parquet(WORK / "train_truth.parquet")
    print("truth", truth.shape)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["train", "test"]:
        prep_split(split)
    prep_truth()
