"""Does the Swing model's (direction_xgboost_v5) performance drift across
the 5-year training window -- e.g. was year 1 an easier/harder regime than
year 5? Answers a direct user question: the existing walk-forward
validation already trains-on-earlier/tests-on-later (so it CAN'T be
fooled by testing on the same period it trained on), but
scripts/tune_v5.py only ever kept the 4-fold AVERAGE
(`pd.DataFrame(fold_ml).mean()`) -- individual fold numbers were logged to
the terminal during training and never saved. This script reruns the
identical walk-forward folds and reports each one SEPARATELY, plus a
per-calendar-year breakdown of the raw (model-free) triple-barrier label
rate, to see whether the opportunity itself -- not just the model -- is
getting more or less common over time.

Uses the exact same panel construction, fold generation, and XGBoost
hyperparameters as scripts/train_v5.py (the actual production model), not
a re-derived approximation.

Usage:
    python -m scripts.check_fold_drift
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    BUY_THRESHOLD,
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

logger = get_logger("scripts.check_fold_drift")

LIVE_THRESHOLD = 0.60  # engine/decision.py's actual live BUY_THRESHOLD, not train_v5's 0.65 reference


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    df, feature_cols = prepare_panel(features, labels)
    logger.info("Panel ready: %d rows, %d features, date range %s to %s",
                len(df), len(feature_cols), df["date"].min(), df["date"].max())

    # --- Part 1: raw (model-free) base rate per calendar year -- is the
    # OPPORTUNITY itself (a stock later hitting +5% before -2.5% within 10
    # days, regardless of any model) becoming more or less common? ---
    df["year"] = pd.to_datetime(df["date"]).dt.year
    logger.info("=" * 70)
    logger.info("PART 1: raw label positive rate per calendar year (model-free)")
    logger.info("=" * 70)
    yearly = df.groupby("year")["label"].agg(n="count", base_rate="mean")
    logger.info("\n%s", yearly.to_string())

    # --- Part 2: model performance per walk-forward fold, NOT averaged ---
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=10)
    logger.info("=" * 70)
    logger.info("PART 2: model performance per walk-forward fold (same folds/params as train_v5.py)")
    logger.info("=" * 70)

    fold_rows = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()

        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("Fold %d: skipped (insufficient data)", fold_i)
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        m65 = ml_metrics(y_test, prob, BUY_THRESHOLD)
        tr65 = trading_metrics(y_test, prob, BUY_THRESHOLD)
        m60 = ml_metrics(y_test, prob, LIVE_THRESHOLD)
        tr60 = trading_metrics(y_test, prob, LIVE_THRESHOLD)

        row = {
            "fold": fold_i,
            "test_start": str(split["test_start_date"])[:10],
            "test_end": str(split["test_end_date"])[:10],
            "n_train": len(X_train), "n_test": len(X_test),
            "test_base_rate": float(y_test.mean()),
            "precision@0.65": m65["precision"], "n_buy@0.65": tr65["n_trades"],
            "win_rate@0.65": tr65["win_rate"], "max_dd@0.65": tr65["max_drawdown_pct"],
            "precision@0.60": m60["precision"], "n_buy@0.60": tr60["n_trades"],
            "win_rate@0.60": tr60["win_rate"], "max_dd@0.60": tr60["max_drawdown_pct"],
            "roc_auc": m65["roc_auc"],
        }
        fold_rows.append(row)
        logger.info(
            "Fold %d [%s -> %s] n_train=%d n_test=%d test_base_rate=%.3f | "
            "@0.65: precision=%.3f n_buy=%d win_rate=%.3f max_dd=%.1f%% | "
            "@0.60: precision=%.3f n_buy=%d win_rate=%.3f max_dd=%.1f%% | AUC=%.3f",
            fold_i, row["test_start"], row["test_end"], row["n_train"], row["n_test"], row["test_base_rate"],
            row["precision@0.65"], row["n_buy@0.65"], row["win_rate@0.65"], row["max_dd@0.65"],
            row["precision@0.60"], row["n_buy@0.60"], row["win_rate@0.60"], row["max_dd@0.60"],
            row["roc_auc"],
        )

    fold_df = pd.DataFrame(fold_rows)
    logger.info("=" * 70)
    logger.info("SUMMARY TABLE (per fold, NOT averaged)")
    logger.info("=" * 70)
    logger.info("\n%s", fold_df.to_string(index=False))

    logger.info("=" * 70)
    logger.info("Spread across folds (max - min) -- a large spread means real drift, not noise:")
    for col in ["precision@0.65", "precision@0.60", "roc_auc", "test_base_rate"]:
        valid = fold_df[col].dropna()
        if len(valid):
            logger.info("  %s: min=%.3f max=%.3f spread=%.3f", col, valid.min(), valid.max(), valid.max() - valid.min())

    fold_df.to_csv("data/fold_drift_results.csv", index=False)
    yearly.to_csv("data/yearly_base_rate.csv")
    logger.info("Saved data/fold_drift_results.csv and data/yearly_base_rate.csv")


if __name__ == "__main__":
    run()
