"""Continuation of the parameter-optimality trilogy (slope windows, MA
choice, hyperparameters, RSI/MFI/CMF/ATR/BB, regime thresholds) into 4
more hand-picked windows the user asked about by name after that first
round: ADX (adx_14, window=14), RVOL (rvol_20, window=20), VWAP
(price_vs_vwap20_pct, window=20), and market structure (window=20,
shared by SIX features at once: higher_high_20d/lower_high_20d/
higher_low_20d/lower_low_20d/distance_to_resistance_pct/distance_to_
support_pct -- features/structure.py's compute_structure()).

NOT covered here, deliberately -- relative_strength_20d_pct/sector_
relative_strength_20d_pct need IHSG + sector composite series
reconstructed (a live yfinance fetch + rebuilding sector composites from
the full price panel), a bigger setup than the others; pattern
similarity (features/pattern_similarity.py's WINDOW=20/HORIZON=10/
CORR_THRESHOLD=0.85) is the cross-ticker O(n_query * n_bank) matmul that
module's own docstring documents taking 541s to score a 150-ticker bank
ONCE -- sweeping multiple candidates across the full ~900-ticker
universe would take hours, disproportionate to every other test in this
trilogy. Both flagged as follow-ups, not attempted here.

Same methodology and combined-recheck safeguard as the rest of the
trilogy: one group at a time (holding the other 43 features + already-
verified windows at the shipped baseline), sweep candidates via pooled
walk-forward (target=10%/stop=5%/horizon=5d, threshold=0.60 fixed),
combine whichever candidate wins each group and re-test before drawing
any conclusion.

RESULT (2026-09-14): current windows NOT beaten -- no change adopted,
same decisive pattern as scripts/test_indicator_windows.py (RSI/MFI/
CMF/ATR/BB). Baseline (ADX=14, RVOL=20, VWAP=20, structure=20): pooled
n=804 wilson_lb=71.25%. Every single one of 18 candidates across all 4
groups scored WORSE than baseline (structure's closest miss: window=30,
-0.71pp) -- the combined-recheck wasn't even triggered, nothing won a
group individually. ADX=14 is Wilder's own default (same family as RSI/
ATR, already found robust); RVOL/VWAP/structure=20 are this project's
own repeated "one trading month" convention rather than any single
external standard, but held up anyway. Bug found and fixed mid-run:
the structure group's first attempt renamed its 4 boolean + 2 distance
columns to "_custom" suffixes like every other group here, but scripts/
train_v5.py's prepare_panel() hardcodes those 4 exact original names
(BOOL_COLS) for a bool->float conversion step -- raised a KeyError.
Fixed by having add_structure_window() overwrite the 6 columns in place
under their original names instead (special-cased in run(), the only
group here that works this way). ADX/RVOL/VWAP/structure windows left
unchanged.

NOT covered here -- see this module's own docstring: relative_strength_
20d_pct/sector_relative_strength_20d_pct (need IHSG + sector composite
reconstruction) and pattern similarity (features/pattern_similarity.py,
too expensive to sweep -- its own docstring documents 541s to score a
150-ticker bank once).

Usage:
    python -m scripts.test_more_window_params
"""
import numpy as np
import pandas as pd
import ta
import xgboost as xgb

from pipeline.db import get_engine
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

logger = get_logger("scripts.test_more_window_params")

ADX_CANDIDATES = [7, 10, 18, 21, 25]      # around the current 14 (Wilder default)
AROUND_20_CANDIDATES = [10, 15, 25, 30]    # around the current 20 -- RVOL/VWAP/structure


def _sorted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_values(["stock_code", "date"]).reset_index(drop=True)


def load_prices_with_volume(prices: pd.DataFrame) -> pd.DataFrame:
    engine = get_engine()
    vol = pd.read_sql(
        "SELECT stock_code, date, volume FROM price_history WHERE source_provider = 'yfinance'",
        engine,
    )
    vol["date"] = pd.to_datetime(vol["date"]).dt.date
    merged = prices.merge(vol, on=["stock_code", "date"], how="left")
    logger.info("Volume merged: %d/%d rows have a volume value", merged["volume"].notna().sum(), len(merged))
    return merged


def add_adx_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    def _adx(g):
        return ta.trend.ADXIndicator(g["high"], g["low"], g["close"], window=window).adx()
    adx = prices.groupby("stock_code", group_keys=False).apply(_adx, include_groups=False)
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = adx.reset_index(drop=True)
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_rvol_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    vol_avg = prices.groupby("stock_code")["volume"].transform(lambda s: s.rolling(window, min_periods=window).mean())
    rvol = prices["volume"] / vol_avg
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = rvol
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_vwap_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    typical_price = (prices["high"] + prices["low"] + prices["close"]) / 3
    pv = typical_price * prices["volume"]
    grouped_pv = pv.groupby(prices["stock_code"]).transform(lambda s: s.rolling(window, min_periods=window).sum())
    grouped_vol = prices.groupby("stock_code")["volume"].transform(lambda s: s.rolling(window, min_periods=window).sum())
    vwap = grouped_pv / grouped_vol
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = (prices["close"] - vwap) / vwap * 100
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_structure_window(features, prices, window, new_cols=None):
    """Unlike every other add_* here, this OVERWRITES the 6 structure
    columns in place under their ORIGINAL names (`new_cols` accepted for
    a consistent call signature but ignored) -- scripts/train_v5.py's
    prepare_panel() hardcodes 4 of these exact names (BOOL_COLS) for a
    bool->float conversion step, so renaming them the way every other
    group's "_custom" columns do breaks that step. Callers must NOT also
    pass these column names to eval_variant's drop_cols (there's nothing
    separate to drop -- the merge below replaces the values directly)."""
    prices = _sorted_prices(prices)
    g = prices.groupby("stock_code")
    rolling_high = g["high"].transform(lambda s: s.rolling(window, min_periods=window).max())
    rolling_low = g["low"].transform(lambda s: s.rolling(window, min_periods=window).min())
    prev_high = rolling_high.groupby(prices["stock_code"]).shift(window)
    prev_low = rolling_low.groupby(prices["stock_code"]).shift(window)

    def _tri(current, previous, op):
        r = op(current, previous).astype(float)
        return r.where(previous.notna() & current.notna())

    new_values = prices[["stock_code", "date"]].copy()
    new_values["higher_high_20d"] = _tri(rolling_high, prev_high, np.greater)
    new_values["lower_high_20d"] = _tri(rolling_high, prev_high, np.less)
    new_values["higher_low_20d"] = _tri(rolling_low, prev_low, np.greater)
    new_values["lower_low_20d"] = _tri(rolling_low, prev_low, np.less)
    new_values["distance_to_resistance_pct"] = (rolling_high - prices["close"]) / prices["close"] * 100
    new_values["distance_to_support_pct"] = (prices["close"] - rolling_low) / prices["close"] * 100

    out = features.drop(columns=STRUCTURE_COLS, errors="ignore")
    out = out.merge(new_values, on=["stock_code", "date"], how="left")
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


def eval_variant(features_variant, labels, drop_cols, label_name):
    panel_input = features_variant.drop(columns=[c for c in drop_cols if c in features_variant.columns])
    df, feature_cols = prepare_panel(panel_input, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    return evaluate(df, feature_cols, splits, label_name)


STRUCTURE_COLS = ["higher_high_20d", "lower_high_20d", "higher_low_20d", "lower_low_20d",
                   "distance_to_resistance_pct", "distance_to_support_pct"]

GROUPS = {
    "adx": (ADX_CANDIDATES, ["adx_14"], add_adx_window),
    "rvol": (AROUND_20_CANDIDATES, ["rvol_20"], add_rvol_window),
    "vwap": (AROUND_20_CANDIDATES, ["price_vs_vwap20_pct"], add_vwap_window),
    "structure": (AROUND_20_CANDIDATES, STRUCTURE_COLS, add_structure_window),
}


def run():
    logger.info("Loading %s and %s...", FEATURES_PATH, PRICES_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)
    logger.info("Loaded %d feature rows, %d price rows", len(features), len(prices))
    prices = load_prices_with_volume(prices)

    results = []
    logger.info("#" * 90)
    logger.info("BASELINE (current shipped feature set, unmodified)")
    logger.info("#" * 90)
    baseline_metrics = eval_variant(features, labels, [], "baseline")
    logger.info("[baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "window": "current", **baseline_metrics})

    best_per_group = {}
    for group, (candidates, cols, add_fn) in GROUPS.items():
        logger.info("#" * 90)
        logger.info("GROUP: %s (replacing %s)", group, cols)
        logger.info("#" * 90)
        best = {"wilson_lb": baseline_metrics["wilson_lb"], "window": None}
        for w in candidates:
            label_name = f"{group}_w{w}"
            if group == "structure":
                # add_structure_window overwrites the 6 columns in place
                # under their original names (prepare_panel hardcodes 4 of
                # them) -- nothing separate to drop afterward.
                variant = add_fn(features, prices, w)
                drop_cols = []
            else:
                new_cols = [f"{c}_custom" for c in cols]
                variant = add_fn(features, prices, w, new_cols[0] if len(cols) == 1 else new_cols)
                drop_cols = cols
            m = eval_variant(variant, labels, drop_cols, label_name)
            logger.info("[%s] window=%d roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                        group, w, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                        m["wilson_lb"] - baseline_metrics["wilson_lb"])
            results.append({"group": group, "window": w, **m})
            if m["wilson_lb"] > best["wilson_lb"]:
                best = {"wilson_lb": m["wilson_lb"], "window": w}
        best_per_group[group] = best["window"]

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS (single-variable sweeps):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_more_window_params_results.csv", index=False)
    logger.info("Saved data/test_more_window_params_results.csv")

    winners = {g: w for g, w in best_per_group.items() if w is not None}
    if not winners:
        logger.info("=" * 100)
        logger.info("No group beat baseline individually -- current windows already hold up, nothing to combine.")
        logger.info("=" * 100)
        return

    logger.info("=" * 100)
    logger.info("Best candidates (to combine and re-check): %s", winners)
    logger.info("=" * 100)
    combined = features
    drop_cols = []
    for group, w in winners.items():
        candidates, cols, add_fn = GROUPS[group]
        if group == "structure":
            combined = add_fn(combined, prices, w)
        else:
            new_cols = [f"{c}_best" for c in cols]
            combined = add_fn(combined, prices, w, new_cols[0] if len(cols) == 1 else new_cols)
            drop_cols.extend(cols)
    combined_metrics = eval_variant(combined, labels, drop_cols, "combined_best")
    logger.info("[combined_best] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                combined_metrics["roc_auc"], combined_metrics["n"], combined_metrics["wins"],
                combined_metrics["precision"], combined_metrics["wilson_lb"],
                combined_metrics["wilson_lb"] - baseline_metrics["wilson_lb"])
    if combined_metrics["wilson_lb"] > baseline_metrics["wilson_lb"]:
        logger.info("Combined variant BEATS baseline -- worth considering for adoption.")
    else:
        logger.info("Combined variant does NOT beat baseline -- individual single-variable "
                     "'wins' above were sampling noise, not a real combined improvement. "
                     "Current windows kept unchanged.")


if __name__ == "__main__":
    run()
