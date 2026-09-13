"""Empirical test prompted by a direct user question ("apakah model kita
sudah memperhitungkan Fibonacci") -- confirmed the answer was NO (no
Fibonacci anywhere in this project: not a model feature, not the Detail
Saham chart, not any screener rule -- features/support_resistance.py's
multi-touch pivot clustering is a different technique, based on actual
repeated price touches, not the geometric ratio projection Fibonacci
retracement is). This tests whether it SHOULD be, rather than assuming
either way -- same "test before build" posture as every other feature
question in this project.

Fibonacci retracement: within a recent swing (here, a trailing
FIB_WINDOW-day high/low -- a fixed rolling window, not a detected swing
pivot, for the same no-lookahead simplicity scripts/test_rally_speed_
feature.py's ATR-normalized return already uses), the classic levels are
23.6/38.2/50/61.8/78.6% retracement between the low and high. Three
candidate features, tested together:
  - fib_position: (close - window_low) / (window_high - window_low) --
    where in the 0-100% range price currently sits (a continuous
    generalization, not tied to any one ratio).
  - dist_to_fib618_pct: % distance from close to the 61.8% level
    specifically -- the "golden ratio" level Fibonacci trading lore
    emphasizes most.
  - near_fib_zone: 1 if close is within FIB_TOLERANCE_PCT of ANY of the 5
    standard levels -- the literal "is this a Fibonacci support/
    resistance level" question, tolerance matched to features/
    support_resistance.py's own DEFAULT_TOLERANCE_PCT for the same idea
    (a multi-touch zone) elsewhere in this project.

Same walk-forward methodology as every other feature-addition test here,
but uses POOLED (trade-weighted) precision + Wilson LB as the PRIMARY
metric from the start -- not fold-averaged, per scripts/
tune_v5_extended.py's finding that fold-averaging can mislead (two
hyperparameter configs that looked like clear winners on that metric
turned out to lose once checked pooled).

RESULT (2026-09-13): no meaningful improvement -- REJECTED, not adopted.
Pooled BUY-zone (prob>=0.65): baseline n=890 wins=722 precision=81.12%
wilson_lb=78.42%; with the 3 Fibonacci columns n=895 wins=723
precision=80.78% wilson_lb=78.07%. Delta: roc_auc +0.0001 (noise),
wilson_lb -0.35pp (slightly WORSE), n_signals +5 (negligible coverage
change). Consistent with Fibonacci retracement's genuinely debated
standing in rigorous backtesting literature -- treated here as a
self-fulfilling-heuristic candidate to test, not assumed useful, and it
did not clear that bar. features/support_resistance.py's multi-touch
pivot clustering (a different technique, based on actual repeated price
touches rather than geometric ratio projection) remains the only
support/resistance method used anywhere in this project.

Usage:
    python -m scripts.test_fibonacci_feature
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

logger = get_logger("scripts.test_fibonacci_feature")

FIB_WINDOW = 50  # trading days -- matches this project's existing AVWAP_WINDOW "recent swing" convention
FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_TOLERANCE_PCT = 1.5  # matches features/support_resistance.py's DEFAULT_TOLERANCE_PCT for "near a level"
NEW_COLS = ["fib_position", "dist_to_fib618_pct", "near_fib_zone"]


def add_fibonacci_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """No lookahead: window_high/window_low are trailing rolling max/min
    (pandas .rolling() defaults to trailing, ending at and including the
    current row) over the last FIB_WINDOW trading days -- only past and
    current data, same convention as every other rolling feature in this
    project."""
    prices = prices.sort_values(["stock_code", "date"]).copy()
    g = prices.groupby("stock_code")
    prices["window_high"] = g["high"].transform(lambda s: s.rolling(FIB_WINDOW, min_periods=FIB_WINDOW).max())
    prices["window_low"] = g["low"].transform(lambda s: s.rolling(FIB_WINDOW, min_periods=FIB_WINDOW).min())
    rng = prices["window_high"] - prices["window_low"]

    prices["fib_position"] = np.where(rng > 0, (prices["close"] - prices["window_low"]) / rng, np.nan)

    fib618_price = prices["window_low"] + 0.618 * rng
    prices["dist_to_fib618_pct"] = np.where(rng > 0, (prices["close"] - fib618_price) / prices["close"] * 100, np.nan)

    level_prices = [prices["window_low"] + r * rng for r in FIB_RATIOS]
    dists_pct = pd.concat([(prices["close"] - lp).abs() / prices["close"] * 100 for lp in level_prices], axis=1)
    prices["near_fib_zone"] = np.where(rng > 0, (dists_pct.min(axis=1) <= FIB_TOLERANCE_PCT).astype(float), np.nan)

    merged = features.merge(prices[["stock_code", "date"] + NEW_COLS], on=["stock_code", "date"], how="left")
    return merged


def run_walk_forward_pooled(df, feature_cols, splits, xgb_params, label_name: str):
    pooled_y, pooled_prob = [], []
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
        pooled_y.append(y_test)
        pooled_prob.append(prob)
        logger.info("[%s] fold %d done: n_test=%d", label_name, fold_i, len(X_test))
    if not pooled_y:
        return np.array([]), np.array([])
    return np.concatenate(pooled_y), np.concatenate(pooled_prob)


def part1_pooled_test(features, labels):
    logger.info("#" * 90)
    logger.info("Does fib_position / dist_to_fib618_pct / near_fib_zone help as TRAINING FEATURES?")
    logger.info("#" * 90)

    for c in NEW_COLS:
        logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features), features[c].notna().mean() * 100)

    results = {}
    for variant, include_new in [("A_baseline", False), ("B_with_fibonacci", True)]:
        df, feature_cols = prepare_panel(features, labels)
        if not include_new:
            feature_cols = [c for c in feature_cols if c not in NEW_COLS]
            df = df.drop(columns=NEW_COLS, errors="ignore")
        logger.info("[%s] Panel: %d rows, %d features", variant, len(df), len(feature_cols))

        dates = df["date"].to_numpy()
        splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
        y, prob = run_walk_forward_pooled(df, feature_cols, splits, SWING_XGB_PARAMS, variant)
        roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"] if len(y) else float("nan")

        buy_mask = prob >= BUY_THRESHOLD
        n = int(buy_mask.sum())
        wins = int(y[buy_mask].sum()) if n else 0
        precision = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else 0.0
        results[variant] = {"roc_auc": roc_auc, "n": n, "wins": wins, "precision": precision, "wilson_lb": lb}

    logger.info("=" * 90)
    logger.info("SUMMARY -- POOLED across folds (trade-weighted, not fold-averaged)")
    logger.info("=" * 90)
    for variant, r in results.items():
        logger.info("[%s] roc_auc=%.4f  BUY-zone(prob>=%.2f): n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                     variant, r["roc_auc"], BUY_THRESHOLD, r["n"], r["wins"], r["precision"], r["wilson_lb"])
    a, b = results["A_baseline"], results["B_with_fibonacci"]
    logger.info("-" * 90)
    logger.info("DELTA (B - A): roc_auc %+.4f | wilson_lb %+.4f | n_signals %+d",
                b["roc_auc"] - a["roc_auc"], b["wilson_lb"] - a["wilson_lb"], b["n"] - a["n"])


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_fibonacci_features(features, prices)
    labels = build_panel_labels(prices)
    part1_pooled_test(features, labels)


if __name__ == "__main__":
    run()
