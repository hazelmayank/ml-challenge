"""Train the pair classifier with S1-grouped 2-fold CV and tune the decision threshold.

Decision rule (each S2/S3 record belongs to at most one S1 in the training data):
  a record is assigned to its highest-probability candidate if that probability > t.
Usage: python train.py [sample_fraction]
Outputs: WORK/model.txt, WORK/decision.json, WORK/oof.parquet
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from block import sample_s1_keys
from config import WORK
from features import FEATURES
from metric import macro_f05

MAX_ROUNDS = 800
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=0, verbose=-1)
# BER_LGB_DEVICE=gpu|cuda switches LightGBM to a GPU backend (only if the installed build supports it).
if os.environ.get("BER_LGB_DEVICE", "cpu") != "cpu":
    PARAMS["device_type"] = os.environ["BER_LGB_DEVICE"]


def truth_pairs():
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["k", "entity_id"])
    s23 = pl.read_parquet(WORK / "train_s23.parquet", columns=["k", "entity_id"])
    t = pl.read_parquet(WORK / "train_truth.parquet")
    return (t.join(s23.select(s23k="k", s23_id="entity_id"), on="s23_id")
            .join(s1.select(s1k="k", s1_id="entity_id"), on="s1_id")
            .select("s23k", "s1k"))


def decide(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    best = scored.sort("p", descending=True).unique("s23k", keep="first")
    return best.filter(pl.col("p") > t).select("s23k", "s1k")


def main(fraction=0.25):
    t0 = time.time()
    feat = pl.read_parquet(WORK / "train_cand_sample_feat_parts" / "*.parquet")
    truth = truth_pairs()
    feat = (feat.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["s23k", "s1k"], how="left")
            .with_columns(pl.col("y").fill_null(0))
            .join(truth.rename({"s1k": "true_s1k"}), on="s23k", how="left")
            .with_columns(fold=(pl.coalesce("true_s1k", pl.lit(None, pl.UInt32))
                                .fill_null(pl.col("s23k") + 10**9).hash(3) % 2)))
    print(f"pairs {len(feat):,}  positives {feat['y'].sum():,}  {time.time() - t0:.0f}s")

    # Cast in polars: to_numpy() on mixed dtypes builds a float64 copy (~8 GB for 25M pairs).
    X = feat.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()
    y = feat["y"].to_numpy()
    fold = feat["fold"].to_numpy()
    feat = feat.select("s23k", "s1k", "y")
    print(f"X {X.shape} {X.nbytes / 1e9:.1f} GB", flush=True)
    oof = np.zeros(len(feat), dtype=np.float32)
    iters = []
    for f in (0, 1):
        tr, va = fold != f, fold == f
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=FEATURES, params=PARAMS).construct()
        dva = lgb.Dataset(X[va], y[va], reference=dtr).construct()
        m = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"fold {f}: best_iter {m.best_iteration} {time.time() - t0:.0f}s", flush=True)

    scored = feat.with_columns(p=pl.Series(oof))
    scored.write_parquet(WORK / "oof.parquet")
    keys = sample_s1_keys(fraction)
    ceiling = macro_f05(truth, scored.filter(pl.col("y") == 1).select("s23k", "s1k"), keys)
    print("blocking ceiling:", ceiling)
    best = (0, None)
    for t in np.arange(0.05, 0.96, 0.05):
        r = macro_f05(truth, decide(scored, t), keys)
        print(f"t={t:.2f} {r}")
        if r["macro_f05"] > best[0]:
            best = (r["macro_f05"], float(t))
    print(f"BEST t={best[1]:.2f} macro_f05={best[0]}")

    final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATURES), int(np.mean(iters) * 1.1))
    final.save_model(str(WORK / "model.txt"))
    (WORK / "decision.json").write_text(json.dumps({"threshold": best[1], "cv": best[0]}))
    imp = sorted(zip(FEATURES, final.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(n, round(g / 1e3)) for n, g in imp[:12]])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.25)
