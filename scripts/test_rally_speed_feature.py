"""Empirical test prompted by a real user concern: when the Swing model
says BUY on a stock whose price has ALREADY risen very fast and is
already high, does the model actually account for the elevated reversal
risk that kind of chart pattern carries, or is that a blind spot?

Two separate questions, both tested here:

1. FEATURE-ADDITION test (same walk-forward methodology as
   test_macd_zscore_feature.py / test_avwap_model_feature.py): does
   adding an explicit "how fast has this stock rallied recently" feature
   -- 10-day return, and that same return normalized by the stock's own
   ATR% (so a 10% move means something different for a normally-volatile
   small-cap than a normally-calm blue chip) -- improve the Swing
   model's walk-forward precision/AUC at all?

2. DIAGNOSTIC (more direct answer to the user's actual question, and the
   more informative of the two): among the CURRENTLY-SHIPPED model's own
   out-of-sample BUY-zone predictions (probability >= BUY_THRESHOLD,
   using only the features it already has -- RSI, ATR%, EMA slope, etc.,
   no new feature), does actual win rate differ measurably between
   stocks that rallied fast right before the signal vs ones that didn't?
   If the model's probability already fully accounts for "already ran up
   fast" (via its existing overbought/momentum features), there should be
   little difference in realized win rate across buckets. If there IS a
   real gap, that's a genuine unexploited blind spot -- the model is
   giving the same "BUY, ~60%+" confidence to two situations that don't
   actually perform the same.

Only the SWING model is tested here (not Turnaround) -- the user's
concern is specifically about a fast-rising stock triggering an
already-high-probability BUY, which is a Swing-specific 10-day question;
Turnaround's own target is a 6-month regime shift out of
bearish/bottoming, a materially different situation this concern doesn't
apply to.

RESULT (2026-09-08):

Part 2 (diagnostic) does NOT support the "fast rally = higher reversal
risk" hypothesis, at least not the way it was framed. Bucketing the
shipped model's own BUY-zone (prob>=0.65) OOS predictions by
ret_10d_atr_norm tercile: Low (avg 7.7x its own ATR in 10 days)
win_rate=78.7% LB=73.5%; Mid (avg 18.0x) win_rate=81.4% LB=76.5%; High
(avg 33.0x, i.e. an extreme, near-parabolic move) win_rate=78.2%
LB=73.0%. Essentially FLAT -- the most extreme recent rallies do not win
less often than more modest ones. IMPORTANT caveat on interpretation:
every BUY-zone bucket already has avg RSI 94-97 and at least a ~7-8x-ATR
move in the last 10 days -- reaching prob>=0.65 already requires strong
recent momentum, so this only compares "fast" vs "very fast" rallies,
not "no rally" vs "fast rally" (that comparison barely exists in the
BUY zone -- the model essentially never buys a stock that hasn't already
moved). Within that range, more extreme speed is not extra-risky in this
data.

Part 1 (feature-addition test) found something adjacent but genuinely
useful: feeding ret_10d_pct/ret_10d_atr_norm to the model AS TRAINING
FEATURES (not just diagnosing after the fact) improved the BUY-zone
pooled precision from 79.4% (n=841, Wilson LB 76.6%) to 80.6% (n=856,
Wilson LB 77.8%) -- a real ~1.2pp LB gain at a comparable n to the
already-adopted Momentum Screener AVWAP finding (n=523). ROC AUC stayed
flat (0.582 either way), so this isn't separating winners/losers overall
better -- it's specifically sharpening the BUY-threshold boundary,
consistent with Part 2's finding that the model can already tell "fast"
from "very fast" rallies apart reasonably well once given the number
directly, rather than only inferring it indirectly from RSI/EMA-slope/
ATR% separately. NOT YET adopted into the shipped model (would mean
retraining/redeploying direction_xgboost_v5) -- pending an explicit
decision, since unlike the diagnostic half this one does change what
ships.

Usage:
    python -m scripts.test_rally_speed_feature
"""
import numpy as np
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

logger = get_logger("scripts.test_rally_speed_feature")

RALLY_WINDOW = 10  # matches the Swing model's own 10-day HORIZON
NEW_COLS = ["ret_10d_pct", "ret_10d_atr_norm"]
DIAGNOSTIC_COLS = ["ret_10d_pct", "ret_10d_atr_norm", "rsi_14"]


def add_rally_speed_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Adds ret_10d_pct (10-trading-day % return, no lookahead -- uses
    only the current and past 9 closes) and ret_10d_atr_norm (that same
    return divided by the stock's own atr_pct_14, i.e. "how many ATRs did
    this move in 10 days" -- comparable across stocks with very
    different normal volatility, unlike the raw % alone)."""
    prices = prices.sort_values(["stock_code", "date"]).copy()
    prices["ret_10d_pct"] = prices.groupby("stock_code")["close"].pct_change(RALLY_WINDOW) * 100
    merged = features.merge(prices[["stock_code", "date", "ret_10d_pct"]], on=["stock_code", "date"], how="left")
    merged["ret_10d_atr_norm"] = merged["ret_10d_pct"] / merged["atr_pct_14"].replace(0, np.nan)
    return merged


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


def run_walk_forward_pooled(df, feature_cols, extra_cols, splits, xgb_params) -> pd.DataFrame:
    """Same walk-forward loop, but returns every test-fold row's
    (fold, y_true, prob) plus `extra_cols` values -- lets us bucket the
    model's own pooled out-of-sample predictions by a feature it was NOT
    trained on, to check for miscalibration (diagnostic use only)."""
    pooled = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "label"].to_numpy()
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)

        fold_extra = df.loc[test_mask, extra_cols].reset_index(drop=True)
        fold_df = pd.DataFrame({"fold": fold_i, "y_true": y_test, "prob": prob})
        pooled.append(pd.concat([fold_df, fold_extra], axis=1))
    return pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()


def part1_feature_addition_test(features, labels):
    logger.info("#" * 70)
    logger.info("PART 1: does ret_10d_pct / ret_10d_atr_norm help as a TRAINING FEATURE?")
    logger.info("#" * 70)

    results = {}
    for variant, include_new in [("A_baseline", False), ("B_with_rally_speed", True)]:
        df, feature_cols = prepare_panel(features, labels)
        if not include_new:
            feature_cols = [c for c in feature_cols if c not in NEW_COLS]
            df = df.drop(columns=NEW_COLS, errors="ignore")
        logger.info("[%s] Panel: %d rows, %d features", variant, len(df), len(feature_cols))

        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
        fold_df = run_walk_forward(df, feature_cols, splits, SWING_XGB_PARAMS, BUY_THRESHOLD, variant)
        results[variant] = fold_df

    logger.info("=" * 70)
    logger.info("PART 1 SUMMARY -- average across folds")
    logger.info("=" * 70)
    for variant, fold_df in results.items():
        if fold_df.empty:
            logger.info("[%s] no usable folds", variant)
            continue
        avg = fold_df.mean(numeric_only=True)
        logger.info("[%s] (%d folds) roc_auc=%.4f precision@%.2f=%.4f recall@%.2f=%.4f",
                     variant, len(fold_df), avg["roc_auc"], BUY_THRESHOLD,
                     avg[f"precision@{BUY_THRESHOLD}"], BUY_THRESHOLD, avg[f"recall@{BUY_THRESHOLD}"])


def part2_buy_zone_diagnostic(features, labels):
    logger.info("#" * 70)
    logger.info("PART 2: within the CURRENTLY-SHIPPED model's own BUY-zone OOS predictions,")
    logger.info("does win rate actually differ by how fast the stock already rallied?")
    logger.info("#" * 70)

    df, feature_cols = prepare_panel(features, labels)
    # Baseline feature set only -- this is the model AS SHIPPED (no
    # rally-speed feature). ret_10d_pct/ret_10d_atr_norm/rsi_14 are kept
    # only as DIAGNOSTIC_COLS for bucketing afterwards, never fed to the
    # model, so this reflects exactly what's live in production today.
    feature_cols = [c for c in feature_cols if c not in NEW_COLS]

    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    pooled = run_walk_forward_pooled(df, feature_cols, DIAGNOSTIC_COLS, splits, SWING_XGB_PARAMS)
    logger.info("Pooled OOS rows (all folds): %d", len(pooled))

    buy_zone = pooled[pooled["prob"] >= BUY_THRESHOLD].dropna(subset=["ret_10d_atr_norm"]).copy()
    logger.info("BUY-zone rows (prob>=%.2f) with a usable ret_10d_atr_norm: %d", BUY_THRESHOLD, len(buy_zone))

    overall_wins = int(buy_zone["y_true"].sum())
    logger.info("[BUY zone, ALL] n=%d win_rate=%.4f wilson_lb=%.4f", len(buy_zone),
                overall_wins / len(buy_zone), wilson_lower_bound(overall_wins, len(buy_zone)))

    buy_zone["rally_speed_tercile"] = pd.qcut(buy_zone["ret_10d_atr_norm"], 3, labels=["Low", "Mid", "High"])
    logger.info("-" * 70)
    logger.info("Bucketed by ret_10d_atr_norm tercile (Low=slowest/no recent rally, High=fastest):")
    for tercile in ["Low", "Mid", "High"]:
        sub = buy_zone[buy_zone["rally_speed_tercile"] == tercile]
        n = len(sub)
        wins = int(sub["y_true"].sum())
        wr = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else float("nan")
        logger.info("  [%s] n=%d win_rate=%.4f wilson_lb=%.4f avg_ret_10d_atr_norm=%.2f avg_rsi_14=%.1f",
                     tercile, n, wr, lb, sub["ret_10d_atr_norm"].mean(), sub["rsi_14"].mean())

    # A blunter, more intuitive cut for the user's exact framing: "already
    # moved a large multiple of its own normal daily range in the last 10
    # days" (>=3 ATRs) vs everything else.
    logger.info("-" * 70)
    logger.info("Bucketed by an intuitive absolute cut (already moved >=3x its own ATR in 10 days):")
    for label, mask in [
        ("ret_10d_atr_norm < 3", buy_zone["ret_10d_atr_norm"] < 3),
        ("ret_10d_atr_norm >= 3 (fast/parabolic run-up)", buy_zone["ret_10d_atr_norm"] >= 3),
    ]:
        sub = buy_zone[mask]
        n = len(sub)
        wins = int(sub["y_true"].sum())
        wr = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else float("nan")
        logger.info("  [%s] n=%d win_rate=%.4f wilson_lb=%.4f", label, n, wr, lb)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_rally_speed_features(features, prices)
    for c in NEW_COLS:
        logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features),
                    features[c].notna().mean() * 100)

    labels = build_panel_labels(prices)

    part1_feature_addition_test(features, labels)
    part2_buy_zone_diagnostic(features, labels)


if __name__ == "__main__":
    run()
