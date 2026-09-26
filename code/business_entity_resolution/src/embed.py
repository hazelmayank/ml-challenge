"""GPU embedding candidates: multilingual sentence embeddings of name + address.

Blocking misses are mostly fuzzy/cross-script (Indic names missing from the dictionary,
records without an address, heavy typos, domains), which word tokens cannot catch.
Every record's raw "name | address" is embedded with multilingual-e5-small (MIT). For each
S2/S3 record the EMB_K nearest S1 records of the same country are added to the blocking
candidates, and every candidate pair gets:
  emb_sim  cosine similarity
  emb_gap  emb_sim minus the best emb_sim among the record's candidates
  erank    rank among the record's EMB_K nearest S1 (0 = not among them)
Embedding-only pairs get bscore 0, n_shared 0, brank TOP_K + 1.

Usage: python embed.py train|test [--sample FRACTION] [--probe N]
Input/output: WORK/{split}_cand[_sample].parquet. The blocking-only file is kept as
..._block.parquet so a rerun starts from it. --probe processes the first N S2/S3 records
and prints recall (block vs block + embeddings) without writing the candidate file.
"""
import argparse
import os
import shutil
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from block import TOP_K, recall, sample_s23_keys
from config import WORK, parts_dir

MODEL = os.environ.get("BER_EMB_MODEL", "intfloat/multilingual-e5-small")
EMB_K = int(os.environ.get("BER_EMB_K", 10))
MAX_LEN = 64
BATCH = 1024      # texts per encoder batch (split across GPUs)
CHUNK = 200_000   # S2/S3 records per spilled part
Q_BATCH = 1024    # queries per similarity matmul
PAIR_BATCH = 500_000
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"   # CPU only for local testing
DTYPE = torch.float16 if DEV != "cpu" else torch.float32


class Encoder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).to(DEV, DTYPE).eval()
        self.dim = model.config.hidden_size
        self.model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        print(f"{MODEL} on {DEV} ({torch.cuda.device_count()} GPU(s))", flush=True)

    @torch.inference_mode()
    def __call__(self, texts: list) -> torch.Tensor:
        """L2-normalised mean-pooled embeddings on DEV, in input order."""
        t0 = time.time()
        order = np.argsort([len(t) for t in texts], kind="stable")  # less padding
        out = torch.empty((len(texts), self.dim), dtype=DTYPE, device=DEV)
        for i in range(0, len(texts), BATCH):
            idx = order[i:i + BATCH]
            b = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                         max_length=MAX_LEN, return_tensors="pt").to(DEV)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1)
            out[torch.from_numpy(idx).to(DEV)] = torch.nn.functional.normalize(e.float(), dim=-1).to(DTYPE)
        print(f"  encoded {len(texts):,} in {time.time() - t0:.0f}s "
              f"({len(texts) / max(time.time() - t0, 1e-9):,.0f}/s)", flush=True)
        return out


def texts(df: pl.DataFrame) -> list:
    return df.select(pl.concat_str(pl.lit("query: "), pl.col("business_name").fill_null(""),
                                   pl.lit(" | "), pl.col("business_address").fill_null(""))
                     ).to_series().to_list()


@torch.inference_mode()
def knn(q: torch.Tensor, keys: torch.Tensor, key_ids: torch.Tensor):
    """Top-EMB_K keys per query row -> (row, key id, rank) numpy arrays."""
    k = min(EMB_K, len(key_ids))
    rows, ids = [], []
    for i in range(0, len(q), Q_BATCH):
        top = (q[i:i + Q_BATCH] @ keys.T).topk(k, dim=1).indices
        rows.append(torch.arange(i, i + len(top), device=q.device).repeat_interleave(k))
        ids.append(key_ids[top].flatten())
    rank = np.tile(np.arange(1, k + 1, dtype=np.uint8), len(q))
    return torch.cat(rows).cpu().numpy(), torch.cat(ids).cpu().numpy(), rank


def run(split, sample=None, probe=None):
    t0 = time.time()
    name = f"{split}_cand{'_sample' if sample else ''}"
    base = WORK / f"{name}_block.parquet"
    if "emb_sim" not in pl.read_parquet_schema(WORK / f"{name}.parquet"):
        shutil.copy(WORK / f"{name}.parquet", base)   # fresh blocking output
    block = pl.read_parquet(base)

    enc = Encoder()
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet",
                         columns=["k", "country", "business_name", "business_address"]).sort("k")
    assert s1["k"].to_list()[-1] == len(s1) - 1
    e1 = enc(texts(s1))
    by_country = {}
    for c in s1["country"].unique().to_list():
        ids = torch.from_numpy(s1.filter(pl.col("country") == c)["k"].to_numpy().astype(np.int64)).to(DEV)
        by_country[c] = (e1[ids], ids)
    s1_country = s1["country"]
    del s1

    s23 = pl.read_parquet(WORK / f"{split}_s23.parquet",
                          columns=["k", "country", "business_name", "business_address"])
    if sample:
        s23 = s23.filter(pl.col("k").is_in(sample_s23_keys(sample)))
    s23 = s23.sort("k")
    if probe:
        s23 = s23.head(probe)
    # probe parts cover only part of a chunk, so they must never be resumed by a full run
    fp = f"{MODEL} k={EMB_K} n={len(s23)} " + str(block.select(
        pl.len(), pl.col("s23k").sum(), pl.col("s1k").sum(), pl.col("bscore").sum()).row(0))
    tmp = parts_dir(WORK / f"{name}_emb_{'probe_' if probe else ''}parts", fp)
    for lo in range(0, len(s23), CHUNK):
        chunk = s23.slice(lo, CHUNK)
        part = tmp / f"{lo:09d}_{CHUNK}.parquet"
        if part.exists():
            continue
        e23 = enc(texts(chunk))
        found = []
        for c, (keys, key_ids) in by_country.items():
            rows = np.flatnonzero((chunk["country"] == c).to_numpy())
            if len(rows) == 0:
                continue
            r, ids, rank = knn(e23[torch.from_numpy(rows).to(DEV)], keys, key_ids)
            found.append(pl.DataFrame({"s23k": chunk["k"].to_numpy()[rows][r], "s1k": ids,
                                       "erank": rank}))
        emb = pl.concat(found).with_columns(pl.col("s23k").cast(pl.UInt32),
                                            pl.col("s1k").cast(pl.UInt32))
        blk = block.filter(pl.col("s23k").is_between(chunk["k"].min(), chunk["k"].max()))
        u = (blk.join(emb, on=["s23k", "s1k"], how="full", coalesce=True)
             .with_columns(pl.col("bscore").fill_null(0), pl.col("n_shared").fill_null(0),
                           pl.col("brank").fill_null(TOP_K + 1), pl.col("erank").fill_null(0)))
        pos = torch.from_numpy(np.searchsorted(chunk["k"].to_numpy(), u["s23k"].to_numpy())).to(DEV)
        s1k = torch.from_numpy(u["s1k"].to_numpy().astype(np.int64)).to(DEV)
        sim = np.concatenate([(e23[pos[i:i + PAIR_BATCH]].float() * e1[s1k[i:i + PAIR_BATCH]].float())
                              .sum(1).cpu().numpy() for i in range(0, len(u), PAIR_BATCH)])
        u = (u.with_columns(emb_sim=pl.Series(sim, dtype=pl.Float32))
             .with_columns(emb_gap=pl.col("emb_sim") - pl.col("emb_sim").max().over("s23k"))
             .select(block.columns + ["emb_sim", "emb_gap", "erank"]))
        u.write_parquet(part)
        print(f"  {lo + len(chunk):>10,} / {len(s23):,}  pairs {len(u):,} "
              f"(block {len(blk):,})  {time.time() - t0:.0f}s", flush=True)
        del e23

    if probe:
        return probe_report(split, block, s23, tmp)
    cand = pl.read_parquet(tmp / f"*_{CHUNK}.parquet")
    cand.write_parquet(WORK / f"{name}.parquet")
    print(f"wrote {name}.parquet {cand.shape} (block only {len(block):,}) "
          f"{time.time() - t0:.0f}s")
    if split == "train":
        print("recall below counts block + embedding candidates")
        recall(cand, s23.select("k"))


def probe_report(split, block, s23, tmp):
    assert split == "train", "probe needs labels"
    union = pl.read_parquet(tmp / f"*_{CHUNK}.parquet").filter(pl.col("s23k").is_in(s23["k"]))
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "entity_id"])
    meta = pl.read_parquet(WORK / "train_s23.parquet",
                           columns=["k", "entity_id", "script", "addr_missing", "country"])
    t = (pl.read_parquet(WORK / "train_truth.parquet")
         .join(meta.rename({"k": "s23k", "entity_id": "s23_id"}), on="s23_id")
         .join(s1.rename({"k": "s1k", "entity_id": "s1_id"}), on="s1_id")
         .filter(pl.col("s23k").is_in(s23["k"])))
    hit = (t.join(block.select("s23k", "s1k", in_block=pl.lit(True)), on=["s23k", "s1k"], how="left")
           .join(union.select("s23k", "s1k", "erank", in_union=pl.lit(True)), on=["s23k", "s1k"], how="left")
           .with_columns(pl.col("in_block", "in_union").fill_null(False),
                         in_emb=pl.col("erank").fill_null(0) > 0))
    for by in (None, "script", "addr_missing", "country"):
        g = hit.group_by(by) if by else hit.group_by(pl.lit("all").alias("all"))
        print(g.agg(linked=pl.len(), block=pl.col("in_block").mean().round(4),
                    emb_only=pl.col("in_emb").mean().round(4),
                    union=pl.col("in_union").mean().round(4)))
    n_blk = block.filter(pl.col("s23k").is_in(s23["k"])).height
    print(f"pairs per record: block {n_blk / len(s23):.2f}, union {len(union) / len(s23):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--sample", type=float)
    ap.add_argument("--probe", type=int)
    a = ap.parse_args()
    run(a.split, a.sample, a.probe)
