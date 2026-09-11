"""Empirical test: does the Swing model benefit from trend ACCELERATION/
SLOPE features it has never been given, even in a comparable form?

Prompted by a direct user question ("apakah ada lagi yang belum saya
sampaikan" about Swing's feature set) -- checking the model's real
feature_cols against feature_daily's full column list turned up
ema20_slope_5d, ema20_accel_5d, sma50_slope_10d, macd_hist_slope_3d, and
macd_hist_accel_3d: all computed and stored, all currently EXCLUDED from
training via scripts/train_v5.py's ABSOLUTE_SCALE_COLS.

That exclusion is for a real reason, not an oversight: _slope() in
features/technical.py is `(series - series.shift(n)) / n` -- an absolute
Rupiah-per-day (or per-MACD-histogram-unit-per-day) rate, not comparable
across a Rp50 stock and a Rp50,000 one, same problem the existing
ABSOLUTE_SCALE_COLS (obv, net_foreign_flow, raw macd/macd_hist) already
guard against. But unlike macd_hist's LEVEL (z-scored and tested in
scripts/test_macd_zscore_feature.py -- rejected) or macd_hist's raw slope
(structurally excluded, never tested in ANY form), the accel/slope
features here have never been tested in a properly normalized form at
all -- neither the "flat rejected" nor "shipped" bucket applies yet.

This test normalizes each candidate as a %-of-price rate (dividing by
that day's close, matching the style of price_vs_sma50_pct/ret_10d_pct
elsewhere in this project) and runs the SAME walk-forward A/B methodology
as scripts/test_rally_speed_feature.py's Part 1 (which is how ret_10d_pct/
ret_10d_atr_norm got adopted): baseline feature set vs baseline + these 5
new %-normalized columns, pooled precision/recall at the shipped BUY
threshold, ROC-AUC, averaged across the same 5 walk-forward folds.

RESULT (2026-09-11): flat, no meaningful improvement -- unlike the rally-
speed feature's real gain. ROC-AUC 0.5802 -> 0.5810 (+0.0008, noise).
Pooled BUY-zone (prob>=0.65) precision/Wilson LB across all 5 folds:
baseline n=890 wins=722 precision=81.1% wilson_lb=78.4%; with the 5 new
columns n=849 wins=691 precision=81.4% wilson_lb=78.6% -- a +0.2pp LB
move, an order of magnitude smaller than the rally-speed feature's
adopted +1.2pp gain, and well within fold-to-fold noise. Coverage also
came in slightly LOWER with the new features (849 vs 890 BUY signals
across the same test windows), so there's no coverage trade-off being
missed either. Conclusion: normalizing these 5 absolute-scale columns
into %-of-price form does not unlock the signal ABSOLUTE_SCALE_COLS was
withholding -- NOT adopted. The likely reason: rsi_slope_3d/5d and
cmf_slope_5d already give the model comparable slope/momentum
information in properly bounded, cross-sectionally comparable units;
these 5 candidates were largely redundant with what's already there
rather than genuinely new information.

Usage:
    python -m scripts.test_trend_acceleration_feature
"""
import pandas as pd
import xgboost as xgb

from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.train_v5 import (
    BUY_THRESHOLD,
    FEATURES_PATH,
    HORIZON,
    NUM_BOOST_ROUND,
    PRICES_PATH,
)
from scripts.train_v5 import XGB_PARAMS as SWING_XGB_PARAMS
from scripts.train_v5 import (
    build_panel_labels,
    ml_metrics,
    prepare_panel,
    walk_forward_splits,
)

logger = get_logger("scripts.test_trend_acceleration_feature")

# (source absolute-scale column, new %-of-price column)
NORMALIZE = [
    ("ema20_slope_5d", "ema20_slope_5d_pct"),
    ("ema20_accel_5d", "ema20_accel_5d_pct"),
    ("sma50_slope_10d", "sma50_slope_10d_pct"),
    ("macd_hist_slope_3d", "macd_hist_slope_3d_pct"),
    ("macd_hist_accel_3d", "macd_hist_accel_3d_pct"),
]
NEW_COLS = [new for _, new in NORMALIZE]


def add_trend_accel_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """%-of-price version of each absolute-scale trend/momentum
    slope-or-acceleration column -- comparable across stocks regardless
    of price level, unlike the raw Rupiah/MACD-unit rate ABSOLUTE_SCALE_
    COLS excludes. No lookahead: uses only that day's own close, already
    present for every row."""
    merged = features.merge(prices[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
    for src, new in NORMALIZE:
        merged[new] = merged[src] / merged["close"] * 100
    return merged.drop(columns=["close"])


def run_walk_forward(df, feature_cols, splits, xgb_params, threshold, label_name):
    fold_rows = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(set(y_train.tolist())) < 2:
            logger.info("[%s] fold %d: skipped (insufficient data)", label_name, fold_i)
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        m = ml_metrics(y_test, prob, threshold)
        n_wins = int(((prob >= threshold) & (y_test == 1)).sum())  # tp -- for pooled Wilson LB across folds
        row = {
            "fold": fold_i, "n_test": len(X_test),
            "roc_auc": ml_metrics(y_test, prob, 0.5)["roc_auc"],
            f"precision@{threshold}": m["precision"], f"recall@{threshold}": m["recall"],
            f"n_signals@{threshold}": m["n_buy_signals"], "n_wins": n_wins,
        }
        fold_rows.append(row)
        logger.info("[%s] fold %d: n_test=%d roc_auc=%.4f precision@%.2f=%.3f recall@%.2f=%.3f n_signals=%d",
                    label_name, fold_i, len(X_test), row["roc_auc"], threshold, m["precision"],
                    threshold, m["recall"], m["n_buy_signals"])
    return pd.DataFrame(fold_rows)


def part1_feature_addition_test(features, labels):
    logger.info("#" * 80)
    logger.info("Does adding %d %%-normalized trend-accel/slope features help as TRAINING FEATURES?", len(NEW_COLS))
    logger.info("#" * 80)

    results = {}
    for variant, include_new in [("A_baseline", False), ("B_with_trend_accel", True)]:
        df, feature_cols = prepare_panel(features, labels)
        if not include_new:
            feature_cols = [c for c in feature_cols if c not in NEW_COLS]
            df = df.drop(columns=NEW_COLS, errors="ignore")
        logger.info("[%s] Panel: %d rows, %d features", variant, len(df), len(feature_cols))

        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
        fold_df = run_walk_forward(df, feature_cols, splits, SWING_XGB_PARAMS, BUY_THRESHOLD, variant)
        results[variant] = fold_df

    logger.info("=" * 80)
    logger.info("SUMMARY -- average across folds")
    logger.info("=" * 80)
    for variant, fold_df in results.items():
        if fold_df.empty:
            logger.info("[%s] no usable folds", variant)
            continue
        avg = fold_df.mean(numeric_only=True)
        n_signals_total = int(fold_df[f"n_signals@{BUY_THRESHOLD}"].sum())
        n_wins_total = int(fold_df["n_wins"].sum())
        pooled_precision = n_wins_total / n_signals_total if n_signals_total else float("nan")
        pooled_lb = wilson_lower_bound(n_wins_total, n_signals_total) if n_signals_total else 0.0
        logger.info("[%s] (%d folds) roc_auc=%.4f precision@%.2f=%.4f recall@%.2f=%.4f n_signals@%.2f=%.1f",
                     variant, len(fold_df), avg["roc_auc"], BUY_THRESHOLD,
                     avg[f"precision@{BUY_THRESHOLD}"], BUY_THRESHOLD, avg[f"recall@{BUY_THRESHOLD}"],
                     BUY_THRESHOLD, avg[f"n_signals@{BUY_THRESHOLD}"])
        logger.info("[%s] POOLED across folds: n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                     variant, n_signals_total, n_wins_total, pooled_precision, pooled_lb)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_trend_accel_features(features, prices)
    for c in NEW_COLS:
        logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features),
                    features[c].notna().mean() * 100)

    labels = build_panel_labels(prices)
    part1_feature_addition_test(features, labels)


if __name__ == "__main__":
    run()
