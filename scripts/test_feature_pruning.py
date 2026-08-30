"""One-off empirical test: does dropping features with <2% gain
contribution improve, hurt, or not matter for either model's walk-forward
performance? Reuses the exact same panel-prep/split/train pipeline as
train_v5.py / train_turnaround.py, just with a filtered feature_cols list,
so the comparison is apples-to-apples against the already-recorded numbers.

Usage:
    python -m scripts.test_feature_pruning
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from engine.model import load_model_and_metadata
from pipeline.logging_config import get_logger
from scripts.train_turnaround import XGB_PARAMS as TA_PARAMS
from scripts.train_turnaround import BUY_THRESHOLD as TA_THRESHOLD
from scripts.train_v5 import BUY_THRESHOLD as SWING_THRESHOLD
from scripts.train_v5 import FEATURES_PATH, NUM_BOOST_ROUND, XGB_PARAMS as SWING_PARAMS
from scripts.train_v5 import build_panel_labels as swing_labels, ml_metrics, prepare_panel, walk_forward_splits
from scripts.turnaround_labels import HORIZON_TRADING_DAYS as TA_HORIZON, build_turnaround_labels

logger = get_logger("scripts.test_feature_pruning")


def gain_pct(booster, feature_cols):
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values())
    return {f: gain.get(f, 0.0) / total * 100 if total else 0.0 for f in feature_cols}


def run_walk_forward(df, feature_cols, splits, params, threshold, label):
    fold_metrics = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        m = ml_metrics(y_test, prob, threshold)
        fold_metrics.append(m)
        logger.info("[%s] fold %d: precision=%.3f recall=%.3f roc_auc=%.3f n_signals=%d",
                    label, fold_i, m["precision"], m["recall"], m["roc_auc"], m["n_buy_signals"])
    avg = pd.DataFrame(fold_metrics).mean()
    logger.info("[%s] AVERAGE (%d features): precision=%.3f recall=%.3f roc_auc=%.3f",
                label, len(feature_cols), avg["precision"], avg["recall"], avg["roc_auc"])
    return avg


def test_model(model_version, features_source_fn, params, threshold, horizon, label):
    logger.info("=" * 70)
    logger.info("%s", label)
    logger.info("=" * 70)
    booster_full, meta_full = load_model_and_metadata(model_version)
    full_feature_cols = meta_full["feature_cols"]
    gains = gain_pct(booster_full, full_feature_cols)
    pruned_cols = [f for f in full_feature_cols if gains[f] >= 2.0]
    dropped = [f for f in full_feature_cols if gains[f] < 2.0]
    logger.info("Dropping %d/%d features (<2%% gain): %s", len(dropped), len(full_feature_cols), dropped)
    logger.info("Keeping %d features: %s", len(pruned_cols), pruned_cols)

    df, dates, splits = features_source_fn()

    logger.info("--- FULL feature set (%d features) ---", len(full_feature_cols))
    full_avg = run_walk_forward(df, full_feature_cols, splits, params, threshold, f"{label} FULL")

    logger.info("--- PRUNED feature set (%d features, dropped <2%% gain) ---", len(pruned_cols))
    pruned_avg = run_walk_forward(df, pruned_cols, splits, params, threshold, f"{label} PRUNED")

    logger.info("=" * 70)
    logger.info("%s COMPARISON: precision %.3f -> %.3f (%+.3f) | roc_auc %.3f -> %.3f (%+.3f)",
                label, full_avg["precision"], pruned_avg["precision"], pruned_avg["precision"] - full_avg["precision"],
                full_avg["roc_auc"], pruned_avg["roc_auc"], pruned_avg["roc_auc"] - full_avg["roc_auc"])
    logger.info("=" * 70)


def swing_source():
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(FEATURES_PATH.replace("_features.parquet", "_prices.parquet"))
    labels = swing_labels(prices)
    df, _ = prepare_panel(features, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)
    return df, dates, splits


def turnaround_source():
    features = pd.read_parquet(FEATURES_PATH)
    labels = build_turnaround_labels(features).rename(columns={"turnaround_label": "label"})
    df, _ = prepare_panel(features, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=TA_HORIZON)
    return df, dates, splits


if __name__ == "__main__":
    test_model("direction_xgboost_v5", swing_source, SWING_PARAMS, SWING_THRESHOLD, 10, "SWING")
    test_model("turnaround_xgboost_v1", turnaround_source, TA_PARAMS, TA_THRESHOLD, TA_HORIZON, "TURNAROUND")
