"""Empirical test prompted directly by a user question ("apakah bisa
ditambahkan status dari Wyckoff theory dan kita lihat lagi model kita")
-- adds features.wyckoff's phase/spring/upthrust/range-position status as
CANDIDATE Swing training features and checks, via the same walk-forward
methodology as every other feature test in this project, whether they
actually help -- not assumed either way.

Tested against the CURRENTLY SHIPPED Swing config (target=10%/stop=5%/
horizon=5 trading days, adopted 2026-09-13 -- see scripts/train_v5.py),
not the old 5%/2.5%/10d label.

Same walk-forward methodology as scripts/test_fibonacci_feature.py:
POOLED (trade-weighted) precision + Wilson LB as the PRIMARY metric from
the start, not fold-averaged (scripts/tune_v5_extended.py's finding that
fold-averaging can mislead).

4 candidate columns tested together:
  - wyckoff_phase (one-hot: accumulation/distribution/markup/markdown,
    "indeterminate" as the implicit baseline -- most rows)
  - wyckoff_spring: today undercut yesterday's established range low then
    closed back above it (single-bar proxy for Wyckoff's "Spring" test)
  - wyckoff_upthrust: symmetric, above the range high
  - wyckoff_range_position_pct: where in the trailing range price sits
    (0-100), a continuous generalization of the phase categories

RESULT (2026-09-13): REJECTED as a training feature -- not adopted.
Pooled BUY-zone (prob>=0.60): baseline n=762 wins=548 precision=71.92%
wilson_lb=68.62%; with the 7 Wyckoff columns n=921 wins=648
precision=70.36% wilson_lb=67.33%. Delta: roc_auc +0.0021 (essentially
noise), wilson_lb -1.29pp (worse), n_signals +159 (21% MORE signals at
LOWER precision -- the model got looser, not sharper). Consistent with
this project's other rejected technical-analysis-lore features
(Fibonacci retracement, MACD z-score, Anchored VWAP): a real, named
concept doesn't automatically make a good ML feature, and this one
didn't clear the bar either. Kept as a DISPLAY-ONLY status on Detail
Saham instead (app/pages/1_Detail_Saham.py, via app/style.py's
wyckoff_badge) -- same fate as the RapidAPI foreign-flow feature (see
supplementary-data-source-plan.md), informational value without a
demonstrated predictive edge.

Usage:
    python -m scripts.test_wyckoff_feature
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from features.wyckoff import compute_wyckoff_features
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

logger = get_logger("scripts.test_wyckoff_feature")

WYCKOFF_ONEHOT_COLS = ["wyckoff_phase_accumulation", "wyckoff_phase_distribution", "wyckoff_phase_markup", "wyckoff_phase_markdown"]
NEW_COLS = WYCKOFF_ONEHOT_COLS + ["wyckoff_spring", "wyckoff_upthrust", "wyckoff_range_position_pct"]


def add_wyckoff_features(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    wyckoff_df = compute_wyckoff_features(prices)
    phase_dummies = pd.get_dummies(wyckoff_df["wyckoff_phase"], prefix="wyckoff_phase", dtype=float)
    # "indeterminate" deliberately dropped -- it's the majority class (57%
    # of rows), left as the implicit baseline the 4 phase dummies are
    # relative to, same convention as every other one-hot set in this
    # project (e.g. features.regime's "sideways" isn't its own dummy
    # either -- see scripts/train_v5.py's prepare_panel).
    phase_dummies = phase_dummies.drop(columns=["wyckoff_phase_indeterminate"], errors="ignore")
    wyckoff_df = pd.concat([wyckoff_df.drop(columns=["wyckoff_phase"]), phase_dummies], axis=1)

    merged = features.merge(
        wyckoff_df[["stock_code", "date", "wyckoff_spring", "wyckoff_upthrust", "wyckoff_range_position_pct"] + [c for c in phase_dummies.columns]],
        on=["stock_code", "date"], how="left",
    )
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
    logger.info("Does Wyckoff phase/spring/upthrust/range-position help as TRAINING FEATURES?")
    logger.info("#" * 90)

    for c in ["wyckoff_spring", "wyckoff_upthrust", "wyckoff_range_position_pct"] + WYCKOFF_ONEHOT_COLS:
        if c in features.columns:
            logger.info("%s non-null: %d/%d (%.1f%%)", c, features[c].notna().sum(), len(features), features[c].notna().mean() * 100)

    results = {}
    for variant, include_new in [("A_baseline", False), ("B_with_wyckoff", True)]:
        df, feature_cols = prepare_panel(features, labels)
        if not include_new:
            feature_cols = [c for c in feature_cols if c not in NEW_COLS]
            df = df.drop(columns=[c for c in NEW_COLS if c in df.columns], errors="ignore")
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
    a, b = results["A_baseline"], results["B_with_wyckoff"]
    logger.info("-" * 90)
    logger.info("DELTA (B - A): roc_auc %+.4f | wilson_lb %+.4f | n_signals %+d",
                b["roc_auc"] - a["roc_auc"], b["wilson_lb"] - a["wilson_lb"], b["n"] - a["n"])


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    features = add_wyckoff_features(features, prices)
    labels = build_panel_labels(prices)
    part1_pooled_test(features, labels)


if __name__ == "__main__":
    run()
