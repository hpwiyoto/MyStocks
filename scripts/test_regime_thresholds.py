"""Are the regime classification thresholds optimal, or -- as features/
regime.py's own docstring already admits -- "a documented first-pass
definition meant to be validated/refined empirically in Fase 3, not a
final truth"? Direct user follow-up, last of the parameter-optimality
trilogy (slope windows, MA choice, hyperparameters, base indicator
windows): "cek klasifikasi regimes".

Different in kind from the other tests: regime isn't one numeric column,
it's a RULE-BASED CLASSIFIER (features/regime.py's classify_regime)
feeding 7 one-hot dummy columns into the model. Changing one threshold
can shift which rows fall into MULTIPLE regime categories at once (e.g.
the "RSI<40" cutoff is shared by both bearish and bottoming), so a
candidate here means recomputing the WHOLE regime series with that one
threshold changed and regenerating all 7 dummies, not swapping a single
feature.

4 thresholds tested, one at a time (holding the other 43 features + the
other 3 thresholds at the shipped baseline): overextended's RSI cutoff
(70), the bearish/bottoming "weak RSI" cutoff (40, shared by both
regimes), bullish's RSI cutoff (50), and accumulation's Bollinger-width
rank cutoff (0.30). BB_WIDTH_RANK_WINDOW=100 and FLAT_PRICE_BAND_PCT=5.0
are NOT swept here -- scope kept to the 4 most consequential thresholds
rather than every constant in the module, consistent with this
project's grid-search-overfitting caution (features/regime.py's own
docstring) about testing too many knobs against a noisy metric at once.

Same methodology and combined-recheck safeguard as the other 3 tests in
this trilogy.

RESULT (2026-09-14): current thresholds NOT meaningfully beaten -- no
change adopted. Baseline (overextended RSI>70, weak RSI<40, bullish
RSI>=50, accumulation BB-rank<0.30): pooled n=804 wilson_lb=71.25%. 10
of 11 candidates scored WORSE than baseline (-0.03pp to -1.25pp). The
ONE exception, bullish_rsi=55, scored +0.05pp (71.30% vs 71.25%) -- an
order of magnitude smaller than even the "unconfirmed lead" magnitude
this session's hyperparameter research treated with caution (+0.85pp/
+0.89pp for t7_h10/t5_h5), and far inside the fold-to-fold noise this
project has already measured directly (scripts/check_fold_drift.py:
72-88% precision swings between folds on the OLD label). Not treated as
a real finding despite technically "surviving" the combined-recheck (it
was the only candidate to combine, so that check couldn't have
distinguished it from noise here). Conclusion: regime classification
thresholds (features/regime.py) left unchanged -- unlike RSI/MFI/CMF/
ATR/BB's decisive result, this is a "nothing clearly better found"
result rather than "decisively already optimal", consistent with
regime.py's own docstring calling itself "a documented first-pass
definition ... not a final truth" -- worth revisiting with a larger
combined grid if the model's overall walk-forward sample size grows.

Usage:
    python -m scripts.test_regime_thresholds
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

logger = get_logger("scripts.test_regime_thresholds")

BB_WIDTH_RANK_WINDOW = 100
FLAT_PRICE_BAND_PCT = 5.0

# defaults match features/regime.py exactly
DEFAULTS = {"overextended_rsi": 70, "weak_rsi": 40, "bullish_rsi": 50, "accumulation_bb_rank": 0.30}
CANDIDATES = {
    "overextended_rsi": [65, 75, 80],
    "weak_rsi": [30, 35, 45],
    "bullish_rsi": [45, 55],
    "accumulation_bb_rank": [0.20, 0.25, 0.40],
}


def classify_regime_variant(features: pd.DataFrame, close: pd.Series, overextended_rsi=70, weak_rsi=40,
                             bullish_rsi=50, accumulation_bb_rank=0.30) -> pd.Series:
    """Same logic as features/regime.py's classify_regime, parameterized
    by the 4 thresholds under test -- kept as a separate copy rather than
    modifying the production module, same convention as every other
    research script here that tests a variant of shipped logic."""
    sma_50, sma_200 = features["sma_50"], features["sma_200"]
    rsi_14, rsi_slope_5d = features["rsi_14"], features["rsi_slope_5d"]
    ema_20 = features["ema_20"]
    price_vs_sma50_pct = features["price_vs_sma50_pct"]
    bb_width_pct = features["bb_width_pct"]
    sma50_slope_positive = features["sma50_slope_10d"] > 0
    bb_rank = bb_width_pct.rolling(BB_WIDTH_RANK_WINDOW).rank(pct=True)

    overextended = (rsi_14 > overextended_rsi) & (price_vs_sma50_pct > 15)
    bearish = (close < sma_50) & (sma_50 < sma_200) & (rsi_14 < weak_rsi) & (rsi_slope_5d <= 0)
    bottoming = (close < sma_50) & (rsi_14 < weak_rsi) & (rsi_slope_5d > 0)
    early_reversal = (close > ema_20) & (ema_20 <= sma_50) & (rsi_14.between(45, 65)) & (rsi_slope_5d > 0)
    bullish = (close > sma_50) & (sma_50 > sma_200) & sma50_slope_positive & (rsi_14 >= bullish_rsi)
    accumulation = (bb_rank < accumulation_bb_rank) & (rsi_slope_5d > 0) & (price_vs_sma50_pct.abs() < FLAT_PRICE_BAND_PCT)

    conditions = [overextended, bearish, bottoming, early_reversal, bullish, accumulation]
    choices = ["overextended", "bearish", "bottoming", "early_reversal", "bullish", "accumulation"]
    regime = pd.Series(np.select(conditions, choices, default="sideways"), index=features.index)

    required = [sma_50, sma_200, rsi_14, ema_20, bb_width_pct]
    has_data = pd.concat(required, axis=1).notna().all(axis=1)
    return regime.where(has_data)


def rebuild_regime_dummies(features: pd.DataFrame, prices: pd.DataFrame, **threshold_overrides) -> pd.DataFrame:
    """Swaps the RAW "regime" categorical column (not pre-made dummies --
    scripts/train_v5.py's prepare_panel() is what one-hot-encodes it,
    from this same column name, during panel prep) for a recomputed
    classification under the given threshold override(s). Must replace
    the string column itself, not pre-build regime_* dummies here --
    prepare_panel would build its OWN set from whatever "regime" column
    it finds and collide with any dummies already present."""
    merged = features.merge(prices[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
    params = {**DEFAULTS, **threshold_overrides}
    new_regime = classify_regime_variant(merged, merged["close"], **params)
    out = features.copy()
    out["regime"] = new_regime.to_numpy()
    return out


def run_walk_forward_pooled(df, feature_cols, splits, xgb_params, label_name):
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


def evaluate(df, feature_cols, splits, label_name):
    y, prob = run_walk_forward_pooled(df, feature_cols, splits, SWING_XGB_PARAMS, label_name)
    roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"] if len(y) else float("nan")
    buy_mask = prob >= BUY_THRESHOLD
    n = int(buy_mask.sum())
    wins = int(y[buy_mask].sum()) if n else 0
    precision = wins / n if n else float("nan")
    lb = wilson_lower_bound(wins, n) if n else 0.0
    return {"roc_auc": roc_auc, "n": n, "wins": wins, "precision": precision, "wilson_lb": lb}


def eval_variant(features_variant, labels, label_name):
    df, feature_cols = prepare_panel(features_variant, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    return evaluate(df, feature_cols, splits, label_name)


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))

    results = []
    logger.info("#" * 90)
    logger.info("BASELINE (current shipped feature set, unmodified)")
    logger.info("#" * 90)
    baseline_metrics = eval_variant(features, labels, "baseline")
    logger.info("[baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "value": "current", **baseline_metrics})

    best_per_group = {}
    for group, candidates in CANDIDATES.items():
        logger.info("#" * 90)
        logger.info("GROUP: %s (default=%s)", group, DEFAULTS[group])
        logger.info("#" * 90)
        best = {"wilson_lb": baseline_metrics["wilson_lb"], "value": None}
        for v in candidates:
            label_name = f"{group}_{v}"
            variant = rebuild_regime_dummies(features, prices, **{group: v})
            m = eval_variant(variant, labels, label_name)
            logger.info("[%s] value=%s roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                        group, v, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                        m["wilson_lb"] - baseline_metrics["wilson_lb"])
            results.append({"group": group, "value": v, **m})
            if m["wilson_lb"] > best["wilson_lb"]:
                best = {"wilson_lb": m["wilson_lb"], "value": v}
        best_per_group[group] = best["value"]

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS (single-variable sweeps):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_regime_thresholds_results.csv", index=False)
    logger.info("Saved data/test_regime_thresholds_results.csv")

    winners = {g: v for g, v in best_per_group.items() if v is not None}
    if not winners:
        logger.info("=" * 100)
        logger.info("No group beat baseline individually -- current regime thresholds already hold up, nothing to combine.")
        logger.info("=" * 100)
        return

    logger.info("=" * 100)
    logger.info("Best candidates (to combine and re-check): %s", winners)
    logger.info("=" * 100)
    combined = rebuild_regime_dummies(features, prices, **winners)
    combined_metrics = eval_variant(combined, labels, "combined_best")
    logger.info("[combined_best] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                combined_metrics["roc_auc"], combined_metrics["n"], combined_metrics["wins"],
                combined_metrics["precision"], combined_metrics["wilson_lb"],
                combined_metrics["wilson_lb"] - baseline_metrics["wilson_lb"])
    if combined_metrics["wilson_lb"] > baseline_metrics["wilson_lb"]:
        logger.info("Combined variant BEATS baseline -- worth considering for adoption.")
    else:
        logger.info("Combined variant does NOT beat baseline -- individual single-variable "
                     "'wins' above were sampling noise, not a real combined improvement. "
                     "Current regime thresholds kept unchanged.")


if __name__ == "__main__":
    run()
