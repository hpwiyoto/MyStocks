"""Hyperparameter grid search + threshold sweep for the turnaround model,
same two-stage approach as scripts/tune_v5.py.

Differences from tune_v5's version:
- No trading_metrics (profit_factor/drawdown) here -- turnaround has no
  target_pct/stop_pct trade structure, just a binary "did it turn around"
  label, so hyperparameter selection is by ROC-AUC alone (still holistic
  in the sense of averaging across folds, just one metric instead of
  three).
- Threshold sweep range extends much higher (up to 0.95) -- the
  turnaround label's base rate is ~82%, so 0.60 (v5's BUY_THRESHOLD)
  barely filters anything; the useful range here is different in kind,
  not just degree.

Usage:
    python -m scripts.tune_turnaround
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_turnaround import BUY_THRESHOLD as REFERENCE_THRESHOLD
from scripts.train_v5 import FEATURES_PATH, ml_metrics, prepare_panel, walk_forward_splits
from scripts.turnaround_labels import HOLD_TRADING_DAYS, HORIZON_TRADING_DAYS, build_turnaround_labels

logger = get_logger("scripts.tune_turnaround")

BASE_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 42}
NUM_BOOST_ROUND = 200

GRID = [
    {"max_depth": 3, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 5, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "eta": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.10, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "subsample": 0.6, "colsample_bytree": 0.6},
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 5, "subsample": 0.6, "colsample_bytree": 0.6},
]

THRESHOLD_RANGE = np.concatenate([np.arange(0.50, 0.90, 0.05), np.arange(0.90, 0.99, 0.02)])


def _load_panel():
    logger.info("Loading %s...", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    labels = build_turnaround_labels(features).rename(columns={"turnaround_label": "label"})
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features", len(df), len(feature_cols))
    return df, feature_cols


def _fold_data(df, feature_cols, splits):
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
        fold_ml = []
        for f in folds:
            booster = xgb.train(params, f["dtrain"], num_boost_round=NUM_BOOST_ROUND)
            prob = booster.predict(f["dtest"])
            fold_ml.append(ml_metrics(f["y_test"], prob, REFERENCE_THRESHOLD))
        avg_ml = pd.DataFrame(fold_ml).mean()
        row = {"config_id": cfg_i, **cfg, "roc_auc": avg_ml["roc_auc"], "precision": avg_ml["precision"],
               "n_signals": avg_ml["n_buy_signals"]}
        results.append(row)
        logger.info("Config %d/%d %s: AUC=%.4f precision=%.3f n_signals=%.0f",
                     cfg_i + 1, len(GRID), cfg, row["roc_auc"], row["precision"], row["n_signals"])
    return pd.DataFrame(results)


def pick_best_by_auc(results: pd.DataFrame) -> int:
    return int(results.sort_values("roc_auc", ascending=False).iloc[0]["config_id"])


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
    base_rate = float(y.mean())

    rows = []
    for t in THRESHOLD_RANGE:
        m = ml_metrics(y, prob, t)
        rows.append({"threshold": round(float(t), 2), "precision": m["precision"],
                      "n_signals": m["n_buy_signals"], "recall": m["recall"],
                      "lift_over_base_rate": m["precision"] / base_rate if m["precision"] == m["precision"] else float("nan")})
        logger.info("threshold=%.2f: precision=%.3f recall=%.3f n_signals=%d lift=%.2fx",
                     t, m["precision"], m["recall"], m["n_buy_signals"],
                     rows[-1]["lift_over_base_rate"] if rows[-1]["lift_over_base_rate"] == rows[-1]["lift_over_base_rate"] else float("nan"))
    logger.info("Pooled base rate this validation set: %.3f", base_rate)
    return pd.DataFrame(rows), base_rate


def run():
    df, feature_cols = _load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600,
                                  label_horizon=HORIZON_TRADING_DAYS)
    folds = _fold_data(df, feature_cols, splits)
    logger.info("%d usable folds", len(folds))

    logger.info("=" * 70)
    logger.info("STAGE 1: hyperparameter grid search (%d configs)", len(GRID))
    logger.info("=" * 70)
    hp_results = hyperparameter_search(folds)
    best_id = pick_best_by_auc(hp_results)
    best_cfg = {k: GRID[best_id][k] for k in GRID[best_id]}
    logger.info("BEST CONFIG (highest avg ROC-AUC): #%d %s", best_id, best_cfg)
    logger.info("\n%s", hp_results.to_string(index=False))

    logger.info("=" * 70)
    logger.info("STAGE 2: threshold sweep with best config (pooled out-of-sample)")
    logger.info("=" * 70)
    thr_results, base_rate = threshold_sweep(folds, best_cfg)
    logger.info("\n%s", thr_results.to_string(index=False))

    with open("data/tune_turnaround_results.json", "w") as f:
        json.dump({
            "hyperparameter_search": hp_results.to_dict(orient="records"),
            "best_config_id": best_id,
            "best_config": best_cfg,
            "threshold_sweep": thr_results.to_dict(orient="records"),
            "pooled_base_rate": base_rate,
            "hold_trading_days": HOLD_TRADING_DAYS,
        }, f, indent=2)
    logger.info("Saved data/tune_turnaround_results.json")


if __name__ == "__main__":
    run()
