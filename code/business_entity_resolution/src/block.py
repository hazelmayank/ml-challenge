"""Candidate generation: IDF-weighted token blocking within country.

Each record is turned into tokens (name words, compact name, address words/numbers),
hashed together with the country. S2/S3 records are joined to S1 records sharing a
token whose S1 document frequency is <= DF_MAX; pair score = sum of token IDF weights.
The TOP_K best S1 records per S2/S3 record are kept.

Usage: python block.py train|test [--sample FRACTION]
Output: WORK/{split}_cand.parquet with s23k, s1k, bscore, brank, n_shared
"""
import argparse
import time

import numpy as np
import polars as pl

from config import WORK

DF_MAX = 150
TOP_K = 10
CHUNK = 100_000


def _words(df, col, prefix):
    return (df.select("k", "country", t=pl.col(col).str.split(" "))
            .explode("t").filter(pl.col("t").str.len_chars() >= 2)
            .with_columns(p=pl.lit(prefix)).unique())


def _pairs(words, prefix, max_words=8):
    """Unordered pairs of a record's words (robust to word-order shuffles)."""
    w = words.with_columns(i=pl.int_range(pl.len()).over("k")).filter(pl.col("i") < max_words)
    j = w.join(w.select("k", t2="t", i2="i"), on="k").filter(pl.col("i") < pl.col("i2"))
    lo = pl.min_horizontal("t", "t2")
    hi = pl.max_horizontal("t", "t2")
    return j.select("k", "country", t=pl.concat_str(lo, hi, separator="+"), p=pl.lit(prefix))


def tokens(df: pl.DataFrame) -> pl.DataFrame:
    """(k, h) pairs: one row per distinct token of each record.

    Singles: name words, address words, compact name.
    Pairs:   name word pairs, address word pairs, name word x address number.
    """
    name = _words(df, "name_core", "n")
    addr = _words(df, "addr_norm", "a")
    nums = addr.filter(pl.col("t").str.contains(r"^\d+$"))
    comp = df.select("k", "country", t=pl.col("name_compact"), p=pl.lit("c"))
    name_num = (name.join(nums.select("k", t2="t"), on="k")
                .select("k", "country", t=pl.concat_str("t", "t2", separator="#"),
                        p=pl.lit("x")))
    out = (pl.concat([name, addr, comp, _pairs(name, "nn"), _pairs(addr, "aa"), name_num])
           .select("k", h=pl.concat_str("country", "p", "t", separator="|").hash())
           .unique())
    return out


def sample_s1_keys(fraction):
    """Deterministic subset of train S1 entities (by row key)."""
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k"])
    return s1.filter((pl.col("k").hash(7) % 1000) < int(fraction * 1000))["k"]


def sample_s23_keys(fraction):
    """All S2/S3 records of the sampled S1 entities + the same fraction of unlinked ones."""
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "entity_id"])
    s23 = pl.read_parquet(WORK / "train_s23.parquet", columns=["k", "entity_id"])
    truth = pl.read_parquet(WORK / "train_truth.parquet")
    linked = (s23.join(truth, left_on="entity_id", right_on="s23_id", how="left")
              .join(s1.select(s1k="k", s1_id="entity_id"), on="s1_id", how="left"))
    keep_s1 = sample_s1_keys(fraction)
    keep = linked.filter(pl.col("s1k").is_in(keep_s1)
                         | (pl.col("s1k").is_null()
                            & ((pl.col("k").hash(7) % 1000) < int(fraction * 1000))))
    return keep["k"]


def run(split, sample=None):
    t0 = time.time()
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet",
                         columns=["k", "country", "name_core", "name_compact", "addr_norm"])
    s1_tok = pl.concat([tokens(s1.slice(i, 300_000)) for i in range(0, len(s1), 300_000)])
    n_s1 = s1.group_by("country").len()
    del s1
    df = s1_tok.group_by("h").len("df")
    s1_tok = s1_tok.join(df.filter(pl.col("df") <= DF_MAX), on="h")
    # IDF weight; the corpus size differs by country but log(N/df) with a global N is fine.
    n_total = int(n_s1["len"].sum())
    s1_tok = s1_tok.with_columns(w=(np.log(n_total) - pl.col("df").log()).cast(pl.Float32)
                                 ).select("h", "k", "w").sort("h")
    print(f"s1 tokens {s1_tok.shape} {time.time() - t0:.0f}s", flush=True)

    s23 = pl.read_parquet(WORK / f"{split}_s23.parquet",
                          columns=["k", "country", "name_core", "name_compact", "addr_norm"])
    if sample:
        s23 = s23.filter(pl.col("k").is_in(sample_s23_keys(sample)))
    # Each chunk's result is written to disk; a rerun skips finished chunks, so a crash
    # (the laptop occasionally runs out of memory) only loses the chunk in progress.
    tmp = WORK / f"{split}_cand{'_sample' if sample else ''}_parts"
    tmp.mkdir(exist_ok=True)
    for start in range(0, len(s23), CHUNK):
        part = tmp / f"{start:09d}_{CHUNK}.parquet"
        if part.exists():
            continue
        chunk = tokens(s23.slice(start, CHUNK)).rename({"k": "s23k"})
        pairs = (chunk.join(s1_tok, on="h")
                 .group_by("s23k", "k")
                 .agg(bscore=pl.col("w").sum(), n_shared=pl.len().cast(pl.UInt16))
                 .rename({"k": "s1k"}))
        top = (pairs.with_columns(brank=pl.col("bscore").rank("ordinal", descending=True)
                                  .over("s23k"))
               .filter(pl.col("brank") <= TOP_K)
               .with_columns(pl.col("brank").cast(pl.UInt8)))
        top.write_parquet(part)
        print(f"  {start + CHUNK:>9,} / {len(s23):,}  pairs {len(pairs):,}  "
              f"{time.time() - t0:.0f}s", flush=True)
        del chunk, pairs, top
    cand = pl.read_parquet(tmp / f"*_{CHUNK}.parquet")
    name = f"{split}_cand{'_sample' if sample else ''}.parquet"
    cand.write_parquet(WORK / name)
    print(f"wrote {name} {cand.shape} {time.time() - t0:.0f}s")
    return cand, s23


def recall(cand, s23):
    """Blocking recall on linked training records."""
    truth = pl.read_parquet(WORK / "train_truth.parquet")
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "entity_id"])
    s23_ids = pl.read_parquet(WORK / "train_s23.parquet", columns=["k", "entity_id"])
    t = (truth.join(s23_ids.rename({"k": "s23k", "entity_id": "s23_id"}), on="s23_id")
         .join(s1.rename({"k": "true_s1k", "entity_id": "s1_id"}), on="s1_id")
         .filter(pl.col("s23k").is_in(s23["k"])))
    hit = t.join(cand, left_on=["s23k", "true_s1k"], right_on=["s23k", "s1k"], how="left")
    print(f"linked records checked: {len(hit):,}")
    print(f"recall@{TOP_K}: {hit['brank'].is_not_null().mean():.4f}")
    for k in (1, 2, 3, 5):
        print(f"recall@{k}: {(hit['brank'] <= k).fill_null(False).mean():.4f}")
    print(f"mean candidates per record: {len(cand) / len(s23):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--sample", type=float)
    a = ap.parse_args()
    c, s = run(a.split, a.sample)
    if a.split == "train":
        recall(c, s)
