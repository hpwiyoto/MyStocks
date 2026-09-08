"""Does training on a SHORTER, ROLLING window of recent history (instead of
ALL available history up to the embargo date) improve the Swing model's
precision -- especially in the weak folds scripts/check_fold_drift.py found?
Directly inspired by a paper the user shared (Roumani, AlSalman & Murphy,
"Effective machine learning estimates of stock market returns using
Taylor-rule inputs") whose robustness checks state plainly: "shorter,
rolling windows better capture time-varying relationships" between
predictors and returns -- a US macro-return context, not ours, but the
underlying claim (recent data may represent the CURRENT regime better than
a multi-year pool that dilutes it with stale patterns) is directly
testable on our own walk-forward folds with no new infrastructure.

Same folds/hyperparameters as scripts/train_v5.py and
scripts/check_fold_drift.py -- ONLY the training set's LOWER bound
(how far back training data reaches) varies between "all history" and a
handful of trailing-window sizes.

RESULT (2026-09-08): INCONCLUSIVE -- NOT adopted. In folds 0-2, shorter
windows were consistently WORSE or tied with all_history (less data,
same regime, so trimming it only removes signal). In fold 3 (the weak
one from check_fold_drift.py), a 36-month window did look better
(precision 79.7% vs all_history's 71.4%, Wilson LB 68.3% vs 57.6%) --
but n_buy was only 49-64 signals in that fold, and going even SHORTER
(24/12 months) collapsed to 58.4%/56.3%, well below baseline. A single
fold's small-n improvement that doesn't replicate across the other
folds, and isn't monotonic in window size, is exactly the kind of
result this project has repeatedly treated as suggestive-not-provable
(see scripts/grid_search_momentum_rules.py's rejection of its own #1
result for the same reason). Contrast with
scripts/test_regime_conditional_threshold.py's finding, which held up
POOLED across every fold at much larger n -- that one shipped, this one
didn't.

Usage:
    python -m scripts.test_rolling_window_size
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    FEATURES_PATH,
    NUM_BOOST_ROUND,
    PRICES_PATH,
    XGB_PARAMS,
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    trading_metrics,
    walk_forward_splits,
)

logger = get_logger("scripts.test_rolling_window_size")

LIVE_THRESHOLD = 0.60
WINDOW_SIZES_DAYS = {
    "all_history": None,
    "36_months": 36 * 30,
    "24_months": 24 * 30,
    "12_months": 12 * 30,
}


def _train_eval(df, feature_cols, split, window_days):
    train_mask = df["date"] <= split["train_embargo_end_date"]
    if window_days is not None:
        window_start = split["train_embargo_end_date"] - pd.Timedelta(days=window_days)
        train_mask &= df["date"] > window_start
    test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
    if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
        return None
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    prob = booster.predict(dtest)
    m = ml_metrics(y_test, prob, LIVE_THRESHOLD)
    tr = trading_metrics(y_test, prob, LIVE_THRESHOLD)
    return {"n_train": len(X_train), "precision": m["precision"], "n_buy": tr["n_trades"], "roc_auc": m["roc_auc"]}


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)
    # train_embargo_end_date comes back as datetime64 already comparable to df["date"]

    logger.info("=" * 70)
    logger.info("Per-fold precision@0.60 by training window size")
    logger.info("=" * 70)
    for fold_i, split in enumerate(splits):
        row_results = {}
        for label, window_days in WINDOW_SIZES_DAYS.items():
            result = _train_eval(df, feature_cols, split, window_days)
            row_results[label] = result
        logger.info("Fold %d [%s -> %s]:", fold_i, str(split["test_start_date"])[:10], str(split["test_end_date"])[:10])
        for label, r in row_results.items():
            if r is None:
                logger.info("  %-12s: skipped (insufficient data)", label)
            else:
                logger.info("  %-12s: n_train=%d precision@0.60=%.3f n_buy=%d AUC=%.3f",
                             label, r["n_train"], r["precision"], r["n_buy"], r["roc_auc"])


if __name__ == "__main__":
    run()
