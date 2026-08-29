"""Hyperparameter grid search + threshold sweep for v5, run directly in
Codespaces (same reasoning as scripts/train_v5.py -- no sklearn here).

Two stages:
1. Hyperparameter grid search: N configs x the same 4 walk-forward folds
   train_v5.py uses, evaluated at the reference threshold (0.60, v4's
   current value) and selected HOLISTICALLY (AUC + profit_factor +
   max_drawdown together) -- not just whichever config has the highest AUC,
   same principle v4's hyperparameter selection used (a higher-AUC config
   with much worse drawdown is an overfitting signature, not a better
   model).
2. Threshold sweep: with the WINNING hyperparameter config, pool every
   fold's out-of-sample predictions together (not per-fold, which would
   have too few trades at the high end of the sweep to be meaningful) and
   sweep threshold 0.30->0.70, same range/methodology v4's threshold_sweep
   used.

Usage:
    python -m scripts.tune_v5
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    BUY_THRESHOLD as REFERENCE_THRESHOLD,
    FEATURES_PATH,
    PRICES_PATH,
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    trading_metrics,
    walk_forward_splits,
)

logger = get_logger("scripts.tune_v5")

BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 42}
NUM_BOOST_ROUND = 200

# 10 configs spanning the same knobs v4's grid search covered: tree depth
# (model capacity), min_child_weight (leaf-level regularization), and
# subsample/colsample (row/column subsampling regularization).
GRID = [
    {"max_depth": 3, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},  # v4/v5-so-far baseline
    {"max_depth": 5, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.10, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "subsample": 0.6, "colsample_bytree": 0.6},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 5, "subsample": 0.6, "colsample_bytree": 0.6},
]

THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.05)


def _load_panel():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features", len(df), len(feature_cols))
    return df, feature_cols


def _fold_data(df, feature_cols, splits):
    """Materialize each fold's train/test arrays ONCE (not once per grid
    config) -- the panel/split logic is identical across all 10 configs, so
    redoing it 10x would be pure waste."""
    folds = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data)", fold_i)
            continue
        folds.append({"fold": fold_i, "dtrain": xgb.DMatrix(X_train, label=y_train),
                       "dtest": xgb.DMatrix(X_test), "y_test": y_test})
    return folds


def hyperparameter_search(folds):
    results = []
    for cfg_i, cfg in enumerate(GRID):
        params = {**BASE_PARAMS, **cfg}
        fold_ml, fold_trading = [], []
        for f in folds:
            booster = xgb.train(params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
            prob = booster.predict(f["dtest"])
            fold_ml.append(ml_metrics(f["y_test"], prob, REFERENCE_THRESHOLD))
            fold_trading.append(trading_metrics(f["y_test"], prob, REFERENCE_THRESHOLD))
        avg_ml = pd.DataFrame(fold_ml).mean()
        avg_trading = pd.DataFrame(fold_trading).mean()
        row = {"config_id": cfg_i, **cfg, "roc_auc": avg_ml["roc_auc"], "precision": avg_ml["precision"],
               "profit_factor": avg_trading["profit_factor"], "max_drawdown_pct": avg_trading["max_drawdown_pct"],
               "n_trades": avg_trading["n_trades"]}
        results.append(row)
        logger.info("Config %d/%d %s: AUC=%.4f precision=%.3f profit_factor=%.2f max_dd=%.1f%% n_trades=%.0f",
                     cfg_i + 1, len(GRID), cfg, row["roc_auc"], row["precision"], row["profit_factor"],
                     row["max_drawdown_pct"], row["n_trades"])
    return pd.DataFrame(results)


def pick_best_holistic(results: pd.DataFrame) -> int:
    """Same holistic principle v4 used: not just highest AUC (which can be
    an overfitting signature if drawdown is much worse) -- rank each of
    AUC/profit_factor/(inverse) max_drawdown, sum the ranks, lowest wins."""
    r = results.copy()
    r["rank_auc"] = r["roc_auc"].rank(ascending=False)
    r["rank_pf"] = r["profit_factor"].rank(ascending=False)
    r["rank_dd"] = r["max_drawdown_pct"].rank(ascending=False)  # less negative = better = higher rank
    r["rank_sum"] = r["rank_auc"] + r["rank_pf"] + r["rank_dd"]
    best = r.sort_values("rank_sum").iloc[0]
    return int(best["config_id"])


def threshold_sweep(folds, best_params):
    params = {**BASE_PARAMS, **best_params}
    pooled_y, pooled_prob = [], []
    for f in folds:
        booster = xgb.train(params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(f["dtest"])
        pooled_y.append(f["y_test"])
        pooled_prob.append(prob)
    y = np.concatenate(pooled_y)
    prob = np.concatenate(pooled_prob)

    rows = []
    for t in THRESHOLD_RANGE:
        m = ml_metrics(y, prob, t)
        tr = trading_metrics(y, prob, t)
        rows.append({"threshold": round(float(t), 2), "precision": m["precision"], "n_trades": tr["n_trades"],
                      "win_rate": tr["win_rate"], "profit_factor": tr["profit_factor"],
                      "max_drawdown_pct": tr["max_drawdown_pct"]})
        logger.info("threshold=%.2f: precision=%.3f n_trades=%d win_rate=%.3f profit_factor=%.2f max_dd=%.1f%%",
                     t, m["precision"], tr["n_trades"], tr["win_rate"], tr["profit_factor"], tr["max_drawdown_pct"])
    return pd.DataFrame(rows)


def run():
    df, feature_cols = _load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)
    folds = _fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    logger.info("=" * 70)
    logger.info("STAGE 1: hyperparameter grid search (%d configs, threshold=%.2f reference)",
                len(GRID), REFERENCE_THRESHOLD)
    logger.info("=" * 70)
    hp_results = hyperparameter_search(folds)
    best_id = pick_best_holistic(hp_results)
    best_cfg = {k: GRID[best_id][k] for k in GRID[best_id]}
    logger.info("BEST CONFIG (holistic): #%d %s", best_id, best_cfg)
    logger.info("\n%s", hp_results.to_string(index=False))

    logger.info("=" * 70)
    logger.info("STAGE 2: threshold sweep with best config (pooled out-of-sample)")
    logger.info("=" * 70)
    thr_results = threshold_sweep(folds, best_cfg)
    logger.info("\n%s", thr_results.to_string(index=False))

    with open("data/tune_v5_results.json", "w") as f:
        json.dump({
            "hyperparameter_search": hp_results.to_dict(orient="records"),
            "best_config_id": best_id,
            "best_config": best_cfg,
            "threshold_sweep": thr_results.to_dict(orient="records"),
        }, f, indent=2)
    logger.info("Saved data/tune_v5_results.json")


if __name__ == "__main__":
    run()
