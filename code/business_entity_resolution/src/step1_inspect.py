"""Step 1: memory-light inspection of the Business Entity Resolution dataset.

Streams every TSV in chunks, then analyses the ground truth. Prints a report
and writes it to <out-dir>/step1_inspection.json. CPU-only; no GPU needed.

Usage (e.g. on a SageMaker notebook / terminal):
    python step1_inspect.py --data-dir /home/ec2-user/SageMaker/student_resource/dataset \
                            --out-dir  /home/ec2-user/SageMaker/reports
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "student_resource" / "dataset"  # overridden by --data-dir
REPORT = ROOT / "reports"  # overridden by --out-dir
CHUNK = 500_000
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
               encoding="utf-8")

PLACEHOLDERS = {"null", "none", "nan", "n/a", "na", "-", "--", "0"}
SCRIPTS = {
    "devanagari": r"[ऀ-ॿ]",
    "tamil": r"[஀-௿]",
    "other_indic": r"[ঀ-୿ఀ-෿]",
    "latin_accented": r"[À-ɏ]",
    "cjk_arabic_cyrillic": r"[Ѐ-ӿ؀-ۿ一-鿿]",
}


def mb(p):
    return round(os.path.getsize(p) / 1e6, 1)


def count_lines(p):
    n = 0
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def inspect_source(path):
    """Stream a source file and accumulate per-column statistics."""
    stats = dict(rows=0, columns=None, country=Counter(), prefix=Counter(),
                 empty=Counter(), placeholder=Counter(), null_token_in_addr=0,
                 scripts={c: Counter() for c in ("business_name", "business_address")},
                 lengths={c: [] for c in ("business_name", "business_address")},
                 samples={})
    ids_numeric = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, **READ_KW):
        stats["columns"] = list(chunk.columns)
        stats["rows"] += len(chunk)
        stats["prefix"].update(chunk["entity_id"].str[:3])
        ids_numeric.append(pd.to_numeric(chunk["entity_id"].str[3:], errors="coerce")
                           .to_numpy(dtype="float64"))
        for country, grp in chunk.groupby("country"):
            stats["country"][country] += len(grp)
            if country not in stats["samples"]:
                stats["samples"][country] = grp.head(3).to_dict("records")
        for col in chunk.columns:
            s = chunk[col].str.strip()
            stats["empty"][col] += int((s == "").sum())
            stats["placeholder"][col] += int(s.str.lower().isin(PLACEHOLDERS).sum())
        stats["null_token_in_addr"] += int(
            chunk["business_address"].str.contains(r"\bnull\b", case=False, regex=True).sum())
        for col in ("business_name", "business_address"):
            s = chunk[col]
            stats["lengths"][col].append(s.str.len().to_numpy())
            for name, pattern in SCRIPTS.items():
                stats["scripts"][col][name] += int(s.str.contains(pattern, regex=True).sum())

    ids = np.concatenate(ids_numeric)
    stats["id_unparseable"] = int(np.isnan(ids).sum())
    stats["id_duplicates"] = int(len(ids) - len(np.unique(ids[~np.isnan(ids)]))
                                 - stats["id_unparseable"])
    for col in ("business_name", "business_address"):
        lens = np.concatenate(stats["lengths"][col])
        stats["lengths"][col] = {k: float(v) for k, v in zip(
            ("min", "p5", "median", "mean", "p95", "max"),
            (lens.min(), np.percentile(lens, 5), np.median(lens), lens.mean(),
             np.percentile(lens, 95), lens.max()))}
    return stats


def load_id_country(path):
    """entity_id (as int) -> country (category): compact enough for 5M rows."""
    df = pd.read_csv(path, usecols=["entity_id", "country"], **READ_KW)
    return pd.DataFrame({"id": df["entity_id"].str[3:].astype("int64"),
                         "country": df["country"].astype("category")})


def analyse_ground_truth():
    gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", **READ_KW)
    out = {"rows": len(gt), "columns": list(gt.columns),
           "duplicate_s1_rows": int(gt["source1_entity_id"].duplicated().sum())}

    lists = gt["matched_entity_ids"].str.strip().str.split(",")
    lists = lists.where(gt["matched_entity_ids"].str.strip() != "", None)
    n_matches = lists.str.len().fillna(0).astype(int)
    out["match_count_distribution"] = {int(k): int(v) for k, v in
                                       n_matches.value_counts().sort_index().items()}
    out["singleton_fraction"] = round(float((n_matches == 0).mean()), 5)
    out["mean_matches_non_singleton"] = round(float(n_matches[n_matches > 0].mean()), 3)

    pairs = pd.DataFrame({"s1": gt["source1_entity_id"], "m": lists}).explode("m").dropna()
    pairs["m"] = pairs["m"].str.strip()
    out["total_true_pairs"] = len(pairs)
    out["duplicate_ids_within_a_list"] = int(pairs.duplicated().sum())
    out["pairs_by_target_source"] = pairs["m"].str[:3].value_counts().to_dict()

    # Does a single S2/S3 record ever belong to more than one S1 entity?
    per_record = pairs["m"].value_counts()
    out["s2s3_records_linked_to_multiple_s1"] = int((per_record > 1).sum())
    out["max_s1_per_s2s3_record"] = int(per_record.max())

    # Composition of match lists (S2 only / S3 only / both)
    has2 = pairs.assign(is2=pairs["m"].str.startswith("S2-"))
    comp = has2.groupby("s1")["is2"].agg(["any", "all"])
    out["list_composition"] = {
        "S2_only": int(comp["all"].sum()),
        "S3_only": int((~comp["any"]).sum()),
        "S2_and_S3": int((comp["any"] & ~comp["all"]).sum()),
    }

    # Country consistency + coverage of S2/S3 records by the ground truth
    s1 = load_id_country(DATA / "train" / "train_source1.tsv")
    gt_ids = set(gt["source1_entity_id"])
    out["s1_ids_missing_from_gt"] = int(
        (~("S1-" + s1["id"].astype(str)).isin(gt_ids)).sum())
    pairs["s1_int"] = pairs["s1"].str[3:].astype("int64")
    pairs["m_int"] = pairs["m"].str[3:].astype("int64")
    pairs["src"] = pairs["m"].str[:2]
    pairs = pairs.merge(s1.rename(columns={"id": "s1_int", "country": "c1"}), on="s1_int",
                        how="left")
    coverage, agree, unknown = {}, 0, 0
    for src in ("S2", "S3"):
        other = load_id_country(DATA / "train" / f"train_source{src[1]}.tsv")
        sub = pairs[pairs["src"] == src].merge(
            other.rename(columns={"id": "m_int", "country": "c2"}), on="m_int", how="left")
        unknown += int(sub["c2"].isna().sum())
        agree += int((sub["c1"].astype(str) == sub["c2"].astype(str)).sum())
        linked = sub["m_int"].nunique()
        coverage[src] = {"records": len(other), "linked_to_some_s1": linked,
                         "fraction_linked": round(linked / len(other), 4)}
        del other, sub
    out["gt_ids_not_found_in_sources"] = unknown
    out["pair_country_agreement"] = round(agree / len(pairs), 5)
    out["s2s3_coverage"] = coverage

    by_country = pd.DataFrame({"c": s1.set_index("id").loc[
        gt["source1_entity_id"].str[3:].astype("int64"), "country"].astype(str).to_numpy(),
        "n": n_matches.to_numpy()})
    out["per_country"] = {c: {"s1": int(len(g)),
                              "singleton_fraction": round(float((g["n"] == 0).mean()), 4),
                              "mean_matches": round(float(g["n"].mean()), 3)}
                          for c, g in by_country.groupby("c")}
    return out


def main():
    t0 = time.time()
    report = {"files": {}}
    for split in ("train", "test"):
        for path in sorted((DATA / split).glob("*.tsv")):
            lines = count_lines(path) - 1
            entry = {"size_mb": mb(path), "data_lines": lines}
            if "source" in path.name:
                entry.update(inspect_source(path))
                entry["rows_match_line_count"] = entry["rows"] == lines
            report["files"][path.name] = entry
            print(f"[{time.time() - t0:6.0f}s] scanned {path.name}", flush=True)

    report["ground_truth"] = analyse_ground_truth()
    print(f"[{time.time() - t0:6.0f}s] ground truth analysed", flush=True)

    REPORT.mkdir(exist_ok=True)
    out = REPORT / "step1_inspection.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="folder containing train/ and test/")
    parser.add_argument("--out-dir", type=Path, default=REPORT)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    args = parser.parse_args()
    DATA, REPORT, CHUNK = args.data_dir, args.out_dir, args.chunk
    main()
