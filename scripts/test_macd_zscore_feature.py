"""Empirical test: does adding MACD, normalized via a per-ticker rolling
z-score, improve the Swing (direction_xgboost_v5) or Turnaround
(turnaround_xgboost_v1) models?

MACD has been on the Detail Saham chart for a while (RSI/MACD/CMF panel),
but was never a training feature in either model -- raw MACD is EMA12-EMA26
in Rupiah, so its magnitude scales with the stock's own price level (same
"absolute-scale" problem scripts/train_v5.py's ABSOLUTE_SCALE_COLS already
excludes obv/net_foreign_flow for). This test never feeds the raw value in;
it z-scores macd/macd_hist against each ticker's own trailing 20-day
mean/std first -- same treatment as obv_zscore_20 (shipped) and
net_foreign_flow_zscore_20 (tested, not shipped -- see
scripts/test_foreign_flow_feature.py). Unlike foreign flow, MACD has full
history coverage (computed from price data present for the whole panel), so
there's no partial-coverage caveat to report here.

Usage:
    python -m scripts.test_macd_zscore_feature
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_turnaround import BUY_THRESHOLD as TURNAROUND_THRESHOLD
from scripts.train_turnaround import XGB_PARAMS as TURNAROUND_XGB_PARAMS
from scripts.train_v5 import (
    BUY_THRESHOLD as SWING_THRESHOLD,
)
from scripts.train_v5 import (
    FEATURES_PATH,
    HORIZON,
    NUM_BOOST_ROUND,
    PRICES_PATH,
)
from scripts.train_v5 import (
    XGB_PARAMS as SWING_XGB_PARAMS,
)
from scripts.train_v5 import (
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)
from scripts.turnaround_labels import HORIZON_TRADING_DAYS as TURNAROUND_HORIZON
from scripts.turnaround_labels import build_turnaround_labels

logger = get_logger("scripts.test_macd_zscore_feature")

ZSCORE_WINDOW = 20
NEW_COLS = ["macd_zscore_20", "macd_hist_zscore_20"]


def add_macd_zscores(features: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    features = features.sort_values(["stock_code", "date"]).copy()
    for src, dst in [("macd", "macd_zscore_20"), ("macd_hist", "macd_hist_zscore_20")]:
        grp = features.groupby("stock_code")[src]
        rolling_mean = grp.transform(lambda s: s.rolling(window, min_periods=5).mean())
        rolling_std = grp.transform(lambda s: s.rolling(window, min_periods=5).std())
        features[dst] = (features[src] - rolling_mean) / rolling_std.replace(0, np.nan)
    return features


def run_walk_forward(df, feature_cols, splits, xgb_params, threshold, label_name):
    fold_rows = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("[%s] fold %d: skipped (insufficient data)", label_name, fold_i)
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        m = ml_metrics(y_test, prob, threshold)
        row = {
            "fold": fold_i, "n_test": len(X_test),
            "roc_auc": ml_metrics(y_test, prob, 0.5)["roc_auc"],
            f"precision@{threshold}": m["precision"], f"recall@{threshold}": m["recall"],
            f"n_signals@{threshold}": m["n_buy_signals"],
        }
        fold_rows.append(row)
        logger.info("[%s] fold %d: n_test=%d roc_auc=%.4f precision@%.2f=%.3f recall@%.2f=%.3f n_signals=%d",
                    label_name, fold_i, len(X_test), row["roc_auc"], threshold, m["precision"],
                    threshold, m["recall"], m["n_buy_signals"])
    return pd.DataFrame(fold_rows)


def test_model(model_name, features, labels, xgb_params, threshold, horizon):
    logger.info("#" * 70)
    logger.info("MODEL: %s", model_name)
    logger.info("#" * 70)

    results = {}
    for variant, include_new in [("A_baseline_no_macd", False), ("B_with_macd_zscore", True)]:
        logger.info("=" * 70)
        logger.info("[%s] VARIANT %s", model_name, variant)
        logger.info("=" * 70)

        df, feature_cols = prepare_panel(features, labels)
        # prepare_panel's ABSOLUTE_SCALE_COLS exclusion already drops raw
        # macd/macd_signal/macd_hist/macd_hist_slope_3d/macd_hist_accel_3d --
        # the new zscore columns aren't excluded by anything, so they're
        # already in feature_cols; drop them for the baseline variant.
        if not include_new:
            feature_cols = [c for c in feature_cols if c not in NEW_COLS]
            df = df.drop(columns=NEW_COLS, errors="ignore")

        logger.info("[%s/%s] Panel: %d rows, %d features", model_name, variant, len(df), len(feature_cols))

        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=horizon)
        fold_df = run_walk_forward(df, feature_cols, splits, xgb_params, threshold, f"{model_name}/{variant}")
        results[variant] = fold_df

    logger.info("=" * 70)
    logger.info("[%s] SUMMARY -- average across folds", model_name)
    logger.info("=" * 70)
    for variant, fold_df in results.items():
        if fold_df.empty:
            logger.info("[%s] no usable folds", variant)
            continue
        avg = fold_df.mean(numeric_only=True)
        logger.info("[%s] (%d folds) roc_auc=%.4f precision@%.2f=%.4f recall@%.2f=%.4f",
                     variant, len(fold_df), avg["roc_auc"], threshold,
                     avg[f"precision@{threshold}"], threshold, avg[f"recall@{threshold}"])
    return results


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_macd_zscores(features)
    for c in NEW_COLS:
        logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features),
                    features[c].notna().mean() * 100)

    swing_labels = build_panel_labels(prices)
    test_model("SWING", features, swing_labels, SWING_XGB_PARAMS, SWING_THRESHOLD, HORIZON)

    turnaround_labels = build_turnaround_labels(features).rename(columns={"turnaround_label": "label"})
    test_model("TURNAROUND", features, turnaround_labels, TURNAROUND_XGB_PARAMS, TURNAROUND_THRESHOLD, TURNAROUND_HORIZON)


if __name__ == "__main__":
    run()
