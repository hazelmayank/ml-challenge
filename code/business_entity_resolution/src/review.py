"""Eyeball test decisions for one country (France has no labels): random accepted matches,
borderline accepts and borderline rejects, with the raw names/addresses side by side.

Usage: python review.py [country] [n]      e.g. python review.py France 25
"""
import json
import sys

import polars as pl

from config import WORK


def main(country="France", n=25):
    cfg_file = WORK / "decision_tuned.json"
    t = json.loads((cfg_file if cfg_file.exists() else WORK / "decision.json").read_text())
    t = t.get("t_country", {}).get(country, t.get("t", t.get("threshold")))
    cols = ["k", "business_name", "business_address", "country"]
    s1 = pl.read_parquet(WORK / "test_s1.parquet", columns=cols).filter(pl.col("country") == country)
    s23 = pl.read_parquet(WORK / "test_s23.parquet", columns=cols[:3])
    best = (pl.read_parquet(WORK / "test_scored.parquet").sort("p", descending=True)
            .unique("s23k", keep="first")
            .join(s1.select("k"), left_on="s1k", right_on="k"))
    pairs = (best.join(s23.rename({"business_name": "name_s23", "business_address": "addr_s23"}),
                       left_on="s23k", right_on="k")
             .join(s1.rename({"business_name": "name_s1", "business_address": "addr_s1"}),
                   left_on="s1k", right_on="k")
             .select(pl.col("p").round(3), "name_s23", "name_s1", "addr_s23", "addr_s1"))
    print(f"{country}: {len(pairs):,} records whose best candidate is a {country} S1; t={t}")
    print(f"accepted {pairs.filter(pl.col('p') > t).height:,}")
    def show(title, df, seed):
        print(f"--- {title} ({len(df):,})")
        print(df.sample(min(n, len(df)), seed=seed))

    with pl.Config(tbl_rows=n, fmt_str_lengths=45, tbl_width_chars=240):
        show("random accepted", pairs.filter(pl.col("p") > t), 1)
        show("borderline accepted", pairs.filter(pl.col("p").is_between(t, t + 0.15)), 2)
        show("borderline rejected", pairs.filter(pl.col("p").is_between(t - 0.25, t)), 3)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "France", int(sys.argv[2]) if len(sys.argv) > 2 else 25)
