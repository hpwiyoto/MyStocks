"""Empirical test: does adding net_foreign_flow (from pipeline/idx_rapidapi_source.py,
backfilled 2026-09-01/02) actually improve the Swing model, per the plan in
.claude/plans/giggly-tickling-eagle.md?

Raw net_foreign_flow is Rupiah-denominated and NOT comparable across stocks
of different sizes (BBCA's daily flow is in the hundreds of billions, a
small-cap's in the millions) -- this is exactly the "absolute-scale feature"
pattern scripts/train_v5.py's ABSOLUTE_SCALE_COLS already excludes for the
same reason (see e91af52: "keluarkan fitur skala-absolut, hasil trading
membaik jelas"). So this test never feeds the raw value in -- it computes a
per-ticker rolling z-score first (net_foreign_flow_zscore_20), the same
normalization already used for obv_zscore_20 in the shipped feature set.

Coverage caveat: the backfill only covers 1 year (2025-09 to 2026-09) of a
5-year panel (2021-2026) -- net_foreign_flow (and its z-score) is NaN for
~80% of all rows. XGBoost handles this natively (NaN = "missing", same as
trailing_pe's ~34%-missing rows already in production), but it also means
early walk-forward folds (testing years before the feature existed) can't
show any effect from it -- only the later fold(s) whose test window falls
inside 2025-09..2026-09 can. Both average-across-folds AND per-fold numbers
are reported so a real effect in later folds isn't hidden by an average
diluted by folds where the feature was simply absent.

Usage:
    python -m scripts.test_foreign_flow_feature
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.train_v5 import (
    FEATURES_PATH,
    HORIZON,
    NUM_BOOST_ROUND,
    PRICES_PATH,
    XGB_PARAMS,
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)

logger = get_logger("scripts.test_foreign_flow_feature")

THRESHOLDS_TO_REPORT = [0.60, 0.65, 0.70]
ZSCORE_WINDOW = 20


def add_foreign_flow_zscore(features: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    features = features.sort_values(["stock_code", "date"]).copy()
    grp = features.groupby("stock_code")["net_foreign_flow"]
    rolling_mean = grp.transform(lambda s: s.rolling(window, min_periods=5).mean())
    rolling_std = grp.transform(lambda s: s.rolling(window, min_periods=5).std())
    features["net_foreign_flow_zscore_20"] = (features["net_foreign_flow"] - rolling_mean) / rolling_std.replace(0, np.nan)
    return features


def run_walk_forward(df, feature_cols, splits, label_name):
    fold_rows = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            logger.info("[%s] fold %d: skipped (insufficient data)", label_name, fold_i)
            continue

        # How much of THIS fold's test window actually has the new feature
        # populated -- the honest context for whether a null result here
        # means "doesn't help" or "wasn't testable yet".
        coverage = float(df.loc[test_mask, "net_foreign_flow_zscore_20"].notna().mean()) if "net_foreign_flow_zscore_20" in feature_cols else None

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        row = {
            "fold": fold_i, "test_start": split["test_start_date"], "test_end": split["test_end_date"],
            "n_test": len(X_test), "ff_coverage_in_test": coverage,
            "roc_auc": ml_metrics(y_test, prob, 0.5)["roc_auc"],
        }
        for thr in THRESHOLDS_TO_REPORT:
            m = ml_metrics(y_test, prob, thr)
            row[f"precision@{thr}"] = m["precision"]
            row[f"recall@{thr}"] = m["recall"]
            row[f"n_signals@{thr}"] = m["n_buy_signals"]
        fold_rows.append(row)
        logger.info(
            "[%s] fold %d (%s..%s, ff_coverage=%s): roc_auc=%.3f precision@0.65=%.3f n_signals@0.65=%d",
            label_name, fold_i, split["test_start_date"], split["test_end_date"],
            f"{coverage:.0%}" if coverage is not None else "n/a",
            row["roc_auc"], row["precision@0.65"], row["n_signals@0.65"],
        )
    return pd.DataFrame(fold_rows)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))
    logger.info("net_foreign_flow non-null: %d/%d (%.1f%%)",
                features["net_foreign_flow"].notna().sum(), len(features),
                features["net_foreign_flow"].notna().mean() * 100)

    features = add_foreign_flow_zscore(features)
    logger.info("net_foreign_flow_zscore_20 non-null: %d/%d (%.1f%%)",
                features["net_foreign_flow_zscore_20"].notna().sum(), len(features),
                features["net_foreign_flow_zscore_20"].notna().mean() * 100)

    labels = build_panel_labels(prices)

    results = {}
    for variant, extra_cols in [("A_baseline_no_foreign_flow", []), ("B_with_foreign_flow_zscore", ["net_foreign_flow_zscore_20"])]:
        logger.info("=" * 70)
        logger.info("VARIANT %s", variant)
        logger.info("=" * 70)

        df, feature_cols = prepare_panel(features, labels)
        # prepare_panel excludes net_foreign_flow (raw) via ABSOLUTE_SCALE_COLS
        # automatically once that's added there; net_foreign_flow_zscore_20
        # isn't excluded by anything, so it's already in feature_cols for
        # variant B and needs removing for variant A (the baseline).
        if not extra_cols:
            feature_cols = [c for c in feature_cols if c != "net_foreign_flow_zscore_20"]
            df = df.drop(columns=["net_foreign_flow_zscore_20"], errors="ignore")

        logger.info("Panel: %d rows, %d features", len(df), len(feature_cols))

        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
        fold_df = run_walk_forward(df, feature_cols, splits, variant)
        results[variant] = fold_df

    logger.info("=" * 70)
    logger.info("SUMMARY -- per fold")
    logger.info("=" * 70)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for variant, fold_df in results.items():
        logger.info("\n[%s]\n%s", variant, fold_df.to_string(index=False))

    logger.info("=" * 70)
    logger.info("SUMMARY -- average across all folds (includes folds with 0%% ff coverage)")
    logger.info("=" * 70)
    for variant, fold_df in results.items():
        avg = fold_df.mean(numeric_only=True)
        logger.info("[%s] roc_auc=%.4f precision@0.65=%.4f recall@0.65=%.4f",
                     variant, avg["roc_auc"], avg["precision@0.65"], avg["recall@0.65"])

    logger.info("=" * 70)
    logger.info("SUMMARY -- average across folds with >0%% ff coverage only (the honest comparison)")
    logger.info("=" * 70)
    for variant, fold_df in results.items():
        covered = fold_df[fold_df["ff_coverage_in_test"].fillna(0) > 0]
        if covered.empty:
            logger.info("[%s] no folds had any foreign-flow coverage", variant)
            continue
        avg = covered.mean(numeric_only=True)
        logger.info("[%s] (%d folds) roc_auc=%.4f precision@0.65=%.4f recall@0.65=%.4f",
                     variant, len(covered), avg["roc_auc"], avg["precision@0.65"], avg["recall@0.65"])


if __name__ == "__main__":
    run()
