"""Empirical test: does adding Anchored VWAP (from the rolling 50-day low)
as a TRAINING FEATURE improve the Swing (direction_xgboost_v5) or
Turnaround (turnaround_xgboost_v1) models?

Context: scripts/test_strategy_6_criteria.py found that requiring
close >= this Anchored VWAP as an extra RULE-based gate improved the
(rule-only, non-ML) Momentum Screener's validated signal (Wilson LB
36.7%->38.3%). That result says nothing about whether the same underlying
information helps the two XGBoost models -- they already see cmf_20,
rvol_20, macd_hist and dozens of other indicators a tree ensemble can
combine on its own, so a hand-crafted gate that helps a fixed rule can
easily be redundant (or even net-negative, if it's a noisier version of
something trees already extract better) once it's just one more feature
among 56. This script tests that directly, same walk-forward methodology
scripts/test_macd_zscore_feature.py and scripts/test_foreign_flow_feature.py
used for their own feature-addition tests.

Unlike the Momentum Screener rule (a hard boolean gate), the ML variant
feeds the model a RELATIVE, scale-free version of the same underlying
signal -- raw avwap_from_low50 is in Rupiah, the same "absolute-scale"
problem scripts/train_v5.py's ABSOLUTE_SCALE_COLS already excludes
macd/obv for. Two engineered columns instead:
  - avwap_dist_pct: (close - avwap_from_low50) / avwap_from_low50 * 100
    -- continuous, same treatment as the already-shipped price_vs_vwap20_pct
    (which uses a rolling, non-anchored 20-day VWAP; this is a different,
    anchored-at-the-recent-low version of the same idea).
  - close_above_avwap: 0/1, the exact boolean the rule-based screener uses.

Volume isn't in data/export_for_colab_prices.parquet (that export only
pulls high/low/close -- see scripts/export_for_colab.py), so this script
queries price_history directly for a fresh stock_code/date/high/low/
close/volume panel instead of reusing PRICES_PATH, then uses THAT for
both label construction (build_panel_labels) and the AVWAP computation,
keeping a single consistent price source.

RESULT (2026-09-08): NOT adopted -- adding avwap_dist_pct/close_above_avwap
as training features makes essentially no difference to either model, and
what little movement there is points slightly negative:
  SWING (5 folds):      baseline roc_auc=0.5806 precision@0.65=0.8040
                         +avwap   roc_auc=0.5793 precision@0.65=0.8036
  TURNAROUND (3 folds): baseline roc_auc=0.6868 precision@0.85=0.9234 recall=0.5526
                         +avwap   roc_auc=0.6869 precision@0.85=0.9219 recall=0.5472
Confirms the hypothesis in this docstring's second paragraph: both models
already see cmf_20, rvol_20, macd_hist, price_vs_vwap20_pct,
distance_to_support/resistance_pct, and regime -- enough correlated
signal that a hand-crafted Anchored-VWAP feature is redundant for a tree
ensemble, unlike for the FIXED rule-based Momentum Screener (which has no
mechanism to combine features on its own, so an explicit extra gate
there was additive -- see scripts/test_strategy_6_criteria.py). Net
lesson: a feature that helps a hard-coded rule doesn't automatically
transfer to an ML model that already has the raw ingredients to
approximate it. Kept as a permanent negative-result artifact, same
practice as test_ihsg_regime_feature.py and test_series_cascade.py.

Usage:
    python -m scripts.test_avwap_model_feature
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.db import get_engine
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

logger = get_logger("scripts.test_avwap_model_feature")

AVWAP_WINDOW = 50
NEW_COLS = ["avwap_dist_pct", "close_above_avwap"]


def load_prices_with_volume() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        """
        SELECT stock_code, date, high, low, close, volume
        FROM price_history
        WHERE source_provider = 'yfinance'
        ORDER BY stock_code, date
        """,
        engine,
    )
    # SQLite (dev) returns `date` as a plain string, not a date object --
    # data/export_for_colab_features.parquet's `date` column is
    # datetime.date (object dtype, from pyarrow's date32 schema). Without
    # this normalization, every merge against `features` on ["stock_code",
    # "date"] silently matches ZERO rows (string "2024-12-05" != date(2024,
    # 12, 5)) rather than raising -- this bit before, so don't remove it.
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Loaded %d price rows across %d tickers (with volume)", len(df), df["stock_code"].nunique())
    return df


def _avwap_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """g: one ticker's rows, ascending by date. Full-history, no-lookahead
    rolling computation -- same approach as
    scripts/test_strategy_6_criteria.py's _add_new_indicators."""
    close = g["close"].to_numpy(dtype=float)
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    volume = g["volume"].to_numpy(dtype=float)
    typical = (high + low + close) / 3
    n = len(close)

    avwap = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - AVWAP_WINDOW + 1)
        seg_close = close[start:i + 1]
        anchor = start + int(np.argmin(seg_close))
        seg_vol = volume[anchor:i + 1]
        vol_sum = seg_vol.sum()
        if vol_sum > 0:
            avwap[i] = (typical[anchor:i + 1] * seg_vol).sum() / vol_sum

    out = g[["stock_code", "date"]].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["avwap_dist_pct"] = (close - avwap) / avwap * 100
    out["close_above_avwap"] = (close >= avwap).astype(float)
    out.loc[np.isnan(avwap), "close_above_avwap"] = np.nan
    return out


def add_avwap_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    avwap_frames = []
    for code, g in prices.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        avwap_frames.append(_avwap_for_ticker(g))
    avwap_df = pd.concat(avwap_frames, ignore_index=True)
    return features.merge(avwap_df, on=["stock_code", "date"], how="left")


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
    for variant, include_new in [("A_baseline_no_avwap", False), ("B_with_avwap", True)]:
        logger.info("=" * 70)
        logger.info("[%s] VARIANT %s", model_name, variant)
        logger.info("=" * 70)

        df, feature_cols = prepare_panel(features, labels)
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
    logger.info("Loading %s and fresh price_history (with volume)...", FEATURES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = load_prices_with_volume()
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_avwap_features(features, prices)
    for c in NEW_COLS:
        logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features),
                    features[c].notna().mean() * 100)

    swing_labels = build_panel_labels(prices)
    test_model("SWING", features, swing_labels, SWING_XGB_PARAMS, SWING_THRESHOLD, HORIZON)

    turnaround_labels = build_turnaround_labels(features).rename(columns={"turnaround_label": "label"})
    test_model("TURNAROUND", features, turnaround_labels, TURNAROUND_XGB_PARAMS, TURNAROUND_THRESHOLD, TURNAROUND_HORIZON)


if __name__ == "__main__":
    run()
