"""Pairwise features for candidate (S2/S3 record, S1 record) pairs.

Usage: python features.py train_cand_sample|test_cand
Output: WORK/{name}_feat.parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import WORK, parts_dir

RECORDS = 100_000  # S2/S3 records per chunk (~1M candidate pairs)
TEXT = ["name_norm", "name_core", "name_compact", "addr_norm", "addr_nums", "addr_ids",
        "hnum", "legal"]
SIMS = {
    "name_norm": [("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio),
                  ("tsort", fuzz.token_sort_ratio), ("jw", JaroWinkler.normalized_similarity)],
    "name_core": [("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio),
                  ("partial", fuzz.partial_ratio)],
    "name_compact": [("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio)],
    "addr_norm": [("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio),
                  ("tsort", fuzz.token_sort_ratio), ("partial", fuzz.partial_ratio)],
    "addr_nums": [("tset", fuzz.token_set_ratio)],
    "addr_ids": [("tset", fuzz.token_set_ratio), ("ratio", fuzz.ratio)],
}
# "Difference detectors": the unlinked S2/S3 records are near-copies of an S1 with one
# detail changed (house number, legal form, a name word), so exact disagreement matters.
DIFF = ["legal_rel", "hnum_rel", "hnum_lev", "ids_a_extra", "ids_b_extra",
        "ids_a_extra_fuzzy", "name_a_extra", "name_b_extra", "first_word_sim"]
FEATURES = (
    # s1_rank / s1_ncand are kept out: the train sample contains only part of each S1's
    # competing records, so their distribution would differ between train and test.
    ["bscore", "n_shared", "brank", "bscore_rel", "bscore_gap",
     "src", "indic", "domain", "addr_missing", "len_name_a", "len_name_b", "len_addr_a",
     "len_addr_b", "num_first_eq"]
    + [f"{c}_{n}" for c, fs in SIMS.items() for n, _ in fs]
    + DIFF
)
# Embedding + stage-1 features from embed.py and prune.py. BER_EMB=0 for runs without a GPU.
EMB = os.environ.get("BER_EMB", "1") != "0"
if EMB:
    FEATURES += ["emb_sim", "emb_gap", "erank", "p1", "p1rank"]   # p1*: prune.py stage-1


def _related(x: str, y: str) -> bool:
    """Same identifier up to truncation/extension (e.g. 1479 vs 14793, 1056 vs 1056c)."""
    return x == y or x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)


def _unmatched_words(words_a, words_b, cutoff=80):
    """Number of words in a with no exact or fuzzy (>= cutoff) counterpart in b."""
    n, set_b = 0, set(words_b)
    for w in words_a:
        if w in set_b:
            continue
        if not any(fuzz.ratio(w, v) >= cutoff for v in words_b):
            n += 1
    return n


DIFF_COLS = ["legal", "legal_1", "hnum", "hnum_1", "addr_ids", "addr_ids_1",
             "name_core", "name_norm_1", "name_core_1", "name_norm"]
WORKERS = max(1, min(8, os.cpu_count() or 1))
_pool = None


def diff_features(pairs: pl.DataFrame) -> dict:
    """Pure-Python per-pair features, spread over a process pool."""
    global _pool
    rows = list(pairs.select(pl.col(c).fill_null("") for c in DIFF_COLS).iter_rows())
    if len(rows) < 50_000:
        return _diff_block(rows)
    if _pool is None:
        import multiprocessing as mp
        _pool = mp.get_context("spawn").Pool(WORKERS)
    step = -(-len(rows) // (WORKERS * 4))
    parts = _pool.map(_diff_block, [rows[i:i + step] for i in range(0, len(rows), step)])
    return {k: np.concatenate([p[k] for p in parts]) for k in DIFF}


def _diff_block(rows) -> dict:
    out = {k: [] for k in DIFF}
    for la, lb, ha, hb, ia, ib, ca, nb, cb, na in rows:
        sa, sb = set(la.split()), set(lb.split())
        out["legal_rel"].append(-1 if not sa or not sb else 1 if sa == sb
                                else 2 if sa <= sb or sb <= sa else 0)
        if not ha or not hb:
            out["hnum_rel"].append(-1)
            out["hnum_lev"].append(-1)
        else:
            out["hnum_rel"].append(1 if ha == hb else 2 if _related(ha, hb)
                                   else 3 if len(ha) == len(hb) else 4)
            out["hnum_lev"].append(Levenshtein.distance(ha, hb))
        ta, tb = ia.split(), ib.split()
        set_ta, set_tb = set(ta), set(tb)
        out["ids_a_extra"].append(len(set_ta - set_tb))
        out["ids_b_extra"].append(len(set_tb - set_ta))
        out["ids_a_extra_fuzzy"].append(
            sum(1 for x in set_ta if not any(_related(x, y) for y in set_tb)))
        wa, wb = ca.split(), cb.split()
        out["name_a_extra"].append(_unmatched_words(wa, nb.split()))
        out["name_b_extra"].append(_unmatched_words(wb, na.split()))
        out["first_word_sim"].append(fuzz.ratio(wa[0], wb[0]) if wa and wb else -1)
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def context(cand: pl.DataFrame) -> pl.DataFrame:
    """Features that compare a pair with the other candidates of the same S2/S3 record."""
    return cand.with_columns(
        bscore_rel=pl.col("bscore") / pl.col("bscore").max().over("s23k"),
        bscore_gap=pl.col("bscore") - pl.col("bscore").filter(pl.col("brank") == 2)
        .first().over("s23k").fill_null(0),
    )


def pair_features(pairs: pl.DataFrame) -> pl.DataFrame:
    out = pairs.with_columns(
        indic=(pl.col("script") == "indic").cast(pl.Int8),
        domain=pl.col("is_domain").cast(pl.Int8),
        addr_missing=pl.col("addr_missing").cast(pl.Int8),
        len_name_a=pl.col("name_norm").str.len_chars().cast(pl.UInt16),
        len_name_b=pl.col("name_norm_1").str.len_chars().cast(pl.UInt16),
        len_addr_a=pl.col("addr_norm").str.len_chars().cast(pl.UInt16),
        len_addr_b=pl.col("addr_norm_1").str.len_chars().cast(pl.UInt16),
        num_first_eq=pl.when(pl.col("addr_nums") == "").then(-1)
        .when(pl.col("addr_nums").str.extract(r"^(\d+)")
              == pl.col("addr_nums_1").str.extract(r"^(\d+)")).then(1)
        .otherwise(0).cast(pl.Int8),
    )
    sims = {}
    for col, fns in SIMS.items():
        a, b = out[col].to_list(), out[f"{col}_1"].to_list()
        for name, fn in fns:
            sims[f"{col}_{name}"] = process.cpdist(a, b, scorer=fn, workers=-1,
                                                   dtype=np.float32)
    sims.update(diff_features(out))
    return out.with_columns(**{k: pl.Series(v) for k, v in sims.items()})


def build(name):
    t0 = time.time()
    split = name.split("_")[0]
    # Work in ranges of S2/S3 keys (all candidates of a record stay together, so the
    # per-record context features are exact) and spill each range to disk. Memory stays
    # flat, and a rerun resumes from the ranges already written.
    cand_scan = pl.scan_parquet(WORK / f"{name}.parquet")
    s23_scan = (pl.scan_parquet(WORK / f"{split}_s23.parquet")
                .select(["k", "src", "script", "is_domain", "addr_missing"] + TEXT)
                .rename({"k": "s23k"}))
    s1 = (pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["k"] + TEXT)
          .rename({c: f"{c}_1" for c in TEXT}).rename({"k": "s1k"}))
    if EMB and "p1" not in cand_scan.collect_schema().names():
        raise SystemExit(f"{name}.parquet has no embedding/pruning columns: run embed.py "
                         "and prune.py first (or set BER_EMB=0)")
    max_k = cand_scan.select(pl.col("s23k").max()).collect().item()
    fp = f"{FEATURES} " + str(cand_scan.select(
        pl.len(), pl.col("s23k").sum(), pl.col("s1k").sum(), pl.col("bscore").sum()).collect().row(0))
    tmp = parts_dir(WORK / f"{name}_feat_parts", fp)
    for lo in range(0, max_k + 1, RECORDS):
        part = tmp / f"{lo:09d}_{RECORDS}.parquet"
        if part.exists():
            continue
        in_range = pl.col("s23k").is_between(lo, lo + RECORDS - 1)
        cand = context(cand_scan.filter(in_range).collect())
        if len(cand) == 0:
            continue
        pairs = (cand.join(s23_scan.filter(in_range).collect(), on="s23k")
                 .join(s1, on="s1k"))
        pair_features(pairs).select(["s23k", "s1k"] + FEATURES).write_parquet(part)
        print(f"  s23k {lo + RECORDS:,}/{max_k + 1:,}  pairs {len(pairs):,} "
              f"{time.time() - t0:.0f}s", flush=True)
        del cand, pairs
    del s1
    # The parts folder is the output (no merged copy: disk space is tight).
    n = pl.scan_parquet(tmp / f"*_{RECORDS}.parquet").select(pl.len()).collect().item()
    print(f"wrote {name}_feat_parts ({n:,} rows) {time.time() - t0:.0f}s")


if __name__ == "__main__":
    build(sys.argv[1])
