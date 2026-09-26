"""Learn an Indic-script word -> Latin word dictionary from the training ground truth.

For true pairs where the S2/S3 name is in an Indic script and has the same number of
words as its (Latin) S1 name, words are aligned by position and co-occurrences counted.
A mapping is kept when it is seen >= MIN_COUNT times and is the dominant translation.
Uses only the provided training data.
Output: WORK/indic_dict.json
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl

from config import DATA, WORK

MIN_COUNT = 2
MIN_SHARE = 0.5
READ = dict(separator="\t", quote_char=None, infer_schema_length=0, encoding="utf8",
            missing_utf8_is_empty_string=True)
INDIC = r"[ऀ-෿]"


def main():
    truth = pl.read_parquet(WORK / "train_truth.parquet")
    s1 = pl.read_csv(DATA / "train" / "train_source1.tsv", **READ,
                     columns=["entity_id", "business_name"])
    s23 = pl.concat([
        pl.read_csv(DATA / "train" / f"train_source{n}.tsv", **READ,
                    columns=["entity_id", "business_name"])
        .filter(pl.col("business_name").str.contains(INDIC)) for n in (2, 3)])
    pairs = (s23.join(truth, left_on="entity_id", right_on="s23_id")
             .join(s1.rename({"entity_id": "s1_id", "business_name": "latin"}), on="s1_id"))
    print(f"indic pairs: {len(pairs):,}")

    counts = defaultdict(Counter)
    for indic, latin in pairs.select("business_name", "latin").iter_rows():
        a, b = indic.split(), re.sub(r"[^\w\s&]", " ", latin.lower()).split()
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if re.search(INDIC, x):
                counts[x][y] += 1
    table = {}
    for word, c in counts.items():
        y, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            table[word] = y
    (WORK / "indic_dict.json").write_text(json.dumps(table, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"learned {len(table):,} word mappings; examples:",
          list(table.items())[:15])


if __name__ == "__main__":
    main()
