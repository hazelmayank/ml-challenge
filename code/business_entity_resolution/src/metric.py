"""Official metric: F0.5 per Source 1 entity, macro-averaged, singletons included."""
import polars as pl


def macro_f05(truth: pl.DataFrame, pred: pl.DataFrame, s1_keys: pl.Series) -> dict:
    """truth/pred: (s23k, s1k) pairs. s1_keys: the S1 entities being evaluated."""
    t = truth.filter(pl.col("s1k").is_in(s1_keys))
    p = pred.filter(pl.col("s1k").is_in(s1_keys))
    tp = t.join(p, on=["s23k", "s1k"]).group_by("s1k").len("tp")
    per = (pl.DataFrame({"s1k": s1_keys})
           .join(t.group_by("s1k").len("nt"), on="s1k", how="left")
           .join(p.group_by("s1k").len("np"), on="s1k", how="left")
           .join(tp, on="s1k", how="left")
           .fill_null(0))
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
         .when((pl.col("nt") == 0) | (pl.col("np") == 0)).then(0.0)
         .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))))
    per = per.with_columns(f=f)
    single = per.filter(pl.col("nt") == 0)
    return {
        "macro_f05": round(per["f"].mean(), 5),
        "singleton_score": round(single["f"].mean(), 4) if len(single) else None,
        "non_singleton_score": round(per.filter(pl.col("nt") > 0)["f"].mean(), 5),
        "precision_pairs": round(len(t.join(p, on=["s23k", "s1k"])) / max(len(p), 1), 5),
        "recall_pairs": round(len(t.join(p, on=["s23k", "s1k"])) / max(len(t), 1), 5),
    }
