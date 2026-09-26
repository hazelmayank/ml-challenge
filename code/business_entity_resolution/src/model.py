"""Pair-classifier backends behind one interface.

BER_MODEL=lgb (default): LightGBM on CPU (BER_LGB_DEVICE=gpu for its OpenCL build).
BER_MODEL=xgb:           XGBoost, on CUDA when a GPU is present (much faster on Kaggle T4s).
The backend used for training is stored in decision.json, so predict.py loads the right one.
"""
import os
import shutil

import numpy as np

KIND = os.environ.get("BER_MODEL", "lgb")
MAX_ROUNDS = 800
EARLY_STOP = 50

LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=0, verbose=-1)
if os.environ.get("BER_LGB_DEVICE", "cpu") != "cpu":
    LGB_PARAMS["device_type"] = os.environ["BER_LGB_DEVICE"]

# lossguide + max_leaves mirrors LightGBM's leaf-wise trees
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  device="cuda" if shutil.which("nvidia-smi") else "cpu",
                  grow_policy="lossguide", max_leaves=63, max_depth=0, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.0)


def path(work, kind=KIND, name="model"):
    return work / (f"{name}.txt" if kind == "lgb" else f"{name}.xgb.ubj")


def fit(X, y, features, X_va=None, y_va=None, rounds=MAX_ROUNDS):
    """Train; with a validation set, early-stop on it. Returns (model, best_iteration)."""
    if KIND == "lgb":
        import lightgbm as lgb
        dtr = lgb.Dataset(X, y, feature_name=features, params=LGB_PARAMS).construct()
        if X_va is None:
            return lgb.train(LGB_PARAMS, dtr, rounds), rounds
        dva = lgb.Dataset(X_va, y_va, reference=dtr).construct()
        m = lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(200)])
        return m, m.best_iteration
    import xgboost as xgb
    print(f"xgboost on {XGB_PARAMS['device']}", flush=True)
    dtr = xgb.QuantileDMatrix(X, y, feature_names=features)
    if X_va is None:
        return xgb.train(XGB_PARAMS, dtr, rounds), rounds
    dva = xgb.QuantileDMatrix(X_va, y_va, feature_names=features, ref=dtr)
    m = xgb.train(XGB_PARAMS, dtr, rounds, evals=[(dva, "valid")],
                  early_stopping_rounds=EARLY_STOP, verbose_eval=200)
    return m, m.best_iteration + 1


def predict(m, X, n_iter=None, kind=KIND) -> np.ndarray:
    if kind == "lgb":
        return m.predict(X, num_iteration=n_iter)
    rng = (0, n_iter) if n_iter else (0, 0)
    return m.inplace_predict(X, iteration_range=rng)


def save(m, work, name="model"):
    m.save_model(str(path(work, name=name)))


def load(work, kind, name="model"):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path(work, kind, name)))
    import xgboost as xgb
    m = xgb.Booster()
    m.load_model(str(path(work, kind, name)))
    if shutil.which("nvidia-smi"):
        m.set_param({"device": "cuda"})
    return m


def importance(m, features, kind=KIND):
    if kind == "lgb":
        return dict(zip(features, m.feature_importance("gain")))
    return m.get_score(importance_type="total_gain")
