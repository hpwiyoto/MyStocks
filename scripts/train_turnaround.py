"""Turnaround model training: same walk-forward/XGBoost infrastructure as
scripts/train_v5.py, different target -- see scripts/turnaround_labels.py
for the label definition (bearish/bottoming -> reaches early_reversal/
bullish and holds >=20 trading days within 6 months).

Only rows where the STARTING regime is bearish/bottoming are candidates;
everything else is unresolved (NaN) and gets dropped by prepare_panel's
existing `label.notna()` filter, same as v5's horizon-unresolved rows.

Usage:
    python -m scripts.train_turnaround
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import FEATURES_PATH, ml_metrics, prepare_panel, walk_forward_splits
from scripts.turnaround_labels import (
    BAD_REGIMES,
    HOLD_TRADING_DAYS,
    HORIZON_TRADING_DAYS,
    build_turnaround_labels,
)

logger = get_logger("scripts.train_turnaround")

MODEL_DIR = "models"
# Tuned via scripts/tune_turnaround.py's pooled out-of-sample sweep: unlike
# v5's swing model (base_rate ~30%, BUY tier lift ~2.5x), this label's base
# rate is already ~82% (most bearish/bottoming stocks DO eventually drift
# into early_reversal/bullish within 6 months), so precision only climbs
# modestly with threshold, and non-monotonically -- it PEAKS at 0.85
# (93.1%) then actually falls at higher thresholds (0.90: 92.5%, 0.96:
# 85.5%, below the pooled base rate) as the selected sample gets small and
# noisy. 0.85 is both the empirical peak and keeps a usable ~13k-signal
# pooled sample (recall 56.9%), not an extreme few-dozen-signal tail.
BUY_THRESHOLD = 0.85

XGB_PARAMS = {
    # max_depth=4, eta=0.03: selected via scripts/tune_turnaround.py's
    # 10-config grid search, highest average walk-forward ROC-AUC (0.686).
    "max_depth": 4, "eta": 0.03, "min_child_weight": 1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "objective": "binary:logistic", "eval_metric": "logloss", "seed": 42,
}
NUM_BOOST_ROUND = 200


def run():
    logger.info("Loading %s...", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    logger.info("Loaded %d feature rows", len(features))

    logger.info("Building turnaround labels (bearish/bottoming -> early_reversal/bullish, "
                "held %d trading days, within %d-day horizon)...", HOLD_TRADING_DAYS, HORIZON_TRADING_DAYS)
    labels = build_turnaround_labels(features)
    resolved = labels["turnaround_label"].notna()
    logger.info("%d (ticker, date) candidate rows (started bearish/bottoming) resolved; "
                "of those, %d turned around (%.1f%%)",
                int(resolved.sum()), int(labels.loc[resolved, "turnaround_label"].sum()),
                labels.loc[resolved, "turnaround_label"].mean() * 100 if resolved.sum() else float("nan"))

    labels = labels.rename(columns={"turnaround_label": "label"})
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows (candidates with a resolved outcome), %d features", len(df), len(feature_cols))

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600,
                                  label_horizon=HORIZON_TRADING_DAYS)
    logger.info("%d walk-forward folds generated", len(splits))

    fold_metrics = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])

        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()

        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data or single-class train set)", fold_i)
            continue

        logger.info("Fold %d: train=%d test=%d train_pos_rate=%.3f test_pos_rate=%.3f",
                    fold_i, len(X_train), len(X_test), y_train.mean(), y_test.mean())

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        m = ml_metrics(y_test, prob, BUY_THRESHOLD)
        fold_metrics.append(m)
        logger.info("Fold %d @ threshold=%.2f: precision=%.3f recall=%.3f roc_auc=%.3f n_signals=%d",
                    fold_i, BUY_THRESHOLD, m["precision"], m["recall"], m["roc_auc"], m["n_buy_signals"])

    logger.info("=" * 70)
    if fold_metrics:
        avg = pd.DataFrame(fold_metrics).mean().to_dict()
        logger.info("AVERAGE ACROSS %d FOLDS (threshold=%.2f): %s", len(fold_metrics), BUY_THRESHOLD,
                    {k: round(v, 4) for k, v in avg.items()})
    else:
        avg = {}
        logger.warning("No folds produced usable results -- not enough resolved candidate rows yet")
    logger.info("=" * 70)

    logger.info("Training final model on all %d rows...", len(df))
    dall = xgb.DMatrix(df[feature_cols], label=df["label"])
    final_booster = xgb.train(XGB_PARAMS, dall, num_boost_round=NUM_BOOST_ROUND)

    base_rate = float(df["label"].mean())
    model_path = f"{MODEL_DIR}/turnaround_xgboost_v1.json"
    meta_path = f"{MODEL_DIR}/turnaround_xgboost_v1_metadata.json"
    final_booster.save_model(model_path)

    metadata = {
        "model_version": "turnaround_xgboost_v1",
        "trained_at": dt.date.today().isoformat(),
        "feature_cols": feature_cols,
        "base_rate": base_rate,
        "starting_regimes": sorted(BAD_REGIMES),
        "target_regimes": ["early_reversal", "bullish"],
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "hold_trading_days": HOLD_TRADING_DAYS,
        "n_training_rows": len(df),
        "tickers": sorted(df["stock_code"].unique().tolist()),
        "hyperparameters": {**XGB_PARAMS, "num_boost_round": NUM_BOOST_ROUND},
        "walk_forward_validation": {
            "n_folds": len(fold_metrics),
            "buy_threshold": BUY_THRESHOLD,
            "avg_ml_metrics": {k: round(float(v), 4) for k, v in avg.items()} if avg else None,
        },
        "notes": (
            "v1 turnaround screener: predicts probability that a stock CURRENTLY in bearish/bottoming "
            "regime reaches early_reversal/bullish and holds there for >=20 trading days within a "
            "6-month horizon, without falling into bearish/bottoming/overextended during the hold "
            "(3rd calibration of the hold rule -- see scripts/turnaround_labels.py's docstring for why "
            "the two stricter/looser versions tried first didn't work). Same feature set and xgb.train "
            "Booster API as direction_xgboost_v5. Hyperparameters (max_depth=4, eta=0.03) and threshold "
            "(0.85) tuned via scripts/tune_turnaround.py, same two-stage grid-search + pooled-sweep "
            "methodology as tune_v5.py. Unlike the swing model, this label's base rate is already ~82%, "
            "so the lift over base rate at the tuned threshold is modest (~1.08x, precision 93.1% vs "
            "82% base) -- this screener's practical value is closer to 'exclude the ~15% likely to "
            "fail' than 'find rare high-upside picks', a genuinely different character from v5's BUY tier."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved %s + %s", model_path, meta_path)
    return metadata


if __name__ == "__main__":
    run()
