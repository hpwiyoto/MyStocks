"""Are the base indicator windows (RSI/CMF/MFI/ATR/BB, NOT their slopes
-- see scripts/test_slope_windows.py -- and NOT the MAs -- see scripts/
test_ma_windows.py) optimal, or just the `ta` library's/Wilder's
conventional defaults never swept for THIS model? Direct user follow-up:
"cek periode dasar RSI/atr/mfi/cmf/bb".

features/technical.py picks a single hand-chosen window per indicator:
RSI=14, MFI=14, ATR=14 (all Wilder's classic default), CMF=20, BB=20
(both `ta` library defaults). Only the DIRECT model features are tested
here: rsi_14, mfi_14, cmf_20, atr_pct_14, bb_width_pct (+bb_width_change_
5d, recomputed together since it's just a 5-day diff of the same series,
not an independently-chosen window the way *_slope_Nd columns are).

Scope boundary, deliberate: rsi_slope_3d/5d, cmf_slope_5d, mfi_slope_5d
are LEFT AT THEIR ORIGINAL 14/20/14-period base indicator (not
recomputed from each candidate window) -- those slope windows were
already tested independently (scripts/test_slope_windows.py, current
windows held up) and mixing that already-settled question into this one
would confound both. Same for ret_10d_atr_norm (left on the original
ATR-14). rsi_distance_50 and bb_width_change_5d/bb_width_pct ARE
recomputed together with their base indicator, since they're not a
separate parameter choice -- just an arithmetic view of the same series.

Recomputed directly from OHLCV (data/export_for_colab_prices.parquet)
using the `ta` library, same functions/conventions as features/
technical.py itself. Same methodology and same combined-recheck
safeguard as the slope-window and MA research: one indicator at a time
(holding the other 43 features at the shipped baseline), sweep
candidates via pooled walk-forward (target=10%/stop=5%/horizon=5d,
threshold=0.60 fixed -- the model's CURRENT threshold, unaffected by the
hyperparameter re-tune since that was default-config-specific and this
test runs on top of it), then combine whichever candidate wins each
group and re-test before drawing any conclusion.

RESULT (2026-09-14): current windows NOT beaten -- no change adopted,
and more decisively than any prior parameter-optimality research this
session (slope windows, MA choice, hyperparameters). Baseline (RSI=14,
MFI=14, CMF=20, ATR=14, BB=20): pooled n=804 wilson_lb=71.25%. Every
single one of the 25 candidates across all 5 indicators scored WORSE
than baseline -- not just the eventual combined check (as with slope/
MA, where individual "wins" existed and only failed once combined), but
every individual candidate too. No combined-recheck was even needed
(the script's own logic skips it when nothing wins a group). Plausible
reason this differs from slope/MA/hyperparameters: RSI=14/MFI=14/ATR=14
(Wilder's own original defaults) and CMF=20/BB=20 (the `ta` library's
defaults, also near-universal market convention) reflect decades of
practitioner convergence across the whole technical-analysis field, not
an ad-hoc choice made once for this project the way slope windows or
the EMA9/20 crossover pair were. Conclusion: RSI/MFI/CMF/ATR/BB windows
left unchanged.

Usage:
    python -m scripts.test_indicator_windows
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

logger = get_logger("scripts.test_indicator_windows")

RSI_MFI_ATR_CANDIDATES = [7, 10, 18, 21, 25]   # around the current 14
CMF_BB_CANDIDATES = [10, 15, 25, 30, 40]        # around the current 20


def _sorted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_values(["stock_code", "date"]).reset_index(drop=True)


def load_prices_with_volume(prices: pd.DataFrame) -> pd.DataFrame:
    """data/export_for_colab_prices.parquet (used everywhere else in this
    project's research scripts) has only stock_code/date/high/low/close --
    no volume, since triple-barrier labeling never needed it. CMF/MFI do,
    so it's pulled from price_history directly here and merged in, once,
    rather than re-queried per candidate window."""
    engine = get_engine()
    vol = pd.read_sql(
        "SELECT stock_code, date, volume FROM price_history WHERE source_provider = 'yfinance'",
        engine,
    )
    vol["date"] = pd.to_datetime(vol["date"]).dt.date
    merged = prices.merge(vol, on=["stock_code", "date"], how="left")
    logger.info("Volume merged: %d/%d rows have a volume value", merged["volume"].notna().sum(), len(merged))
    return merged


def add_rsi_window(features, prices, window, new_cols):
    prices = _sorted_prices(prices)
    rsi = prices.groupby("stock_code")["close"].transform(lambda s: ta.momentum.RSIIndicator(s, window=window).rsi())
    out = prices[["stock_code", "date"]].copy()
    out[new_cols[0]] = rsi          # replaces rsi_14
    out[new_cols[1]] = rsi - 50     # replaces rsi_distance_50
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_mfi_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    def _mfi(g):
        return ta.volume.MFIIndicator(g["high"], g["low"], g["close"], g["volume"], window=window).money_flow_index()
    mfi = prices.groupby("stock_code", group_keys=False).apply(_mfi, include_groups=False)
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = mfi.reset_index(drop=True)
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_cmf_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    def _cmf(g):
        return ta.volume.ChaikinMoneyFlowIndicator(g["high"], g["low"], g["close"], g["volume"], window=window).chaikin_money_flow()
    cmf = prices.groupby("stock_code", group_keys=False).apply(_cmf, include_groups=False)
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = cmf.reset_index(drop=True)
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_atr_window(features, prices, window, new_col):
    prices = _sorted_prices(prices)
    def _atr(g):
        atr = ta.volatility.AverageTrueRange(g["high"], g["low"], g["close"], window=window).average_true_range()
        return atr / g["close"] * 100
    atr_pct = prices.groupby("stock_code", group_keys=False).apply(_atr, include_groups=False)
    out = prices[["stock_code", "date"]].copy()
    out[new_col] = atr_pct.reset_index(drop=True)
    return features.merge(out, on=["stock_code", "date"], how="left")


def add_bb_window(features, prices, window, new_cols):
    prices = _sorted_prices(prices)
    def _bb_width(s):
        bb = ta.volatility.BollingerBands(s, window=window)
        return (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    bb_width = prices.groupby("stock_code")["close"].transform(_bb_width)
    out = prices[["stock_code", "date"]].copy()
    out[new_cols[0]] = bb_width
    out[new_cols[1]] = bb_width - bb_width.groupby(prices["stock_code"]).shift(5)
    return features.merge(out, on=["stock_code", "date"], how="left")


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


GROUPS = {
    "rsi": (RSI_MFI_ATR_CANDIDATES, ["rsi_14", "rsi_distance_50"], add_rsi_window),
    "mfi": (RSI_MFI_ATR_CANDIDATES, ["mfi_14"], add_mfi_window),
    "cmf": (CMF_BB_CANDIDATES, ["cmf_20"], add_cmf_window),
    "atr": (RSI_MFI_ATR_CANDIDATES, ["atr_pct_14"], add_atr_window),
    "bb": (CMF_BB_CANDIDATES, ["bb_width_pct", "bb_width_change_5d"], add_bb_window),
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
            new_cols = [f"{c}_custom" for c in cols]
            if len(cols) == 1:
                variant = add_fn(features, prices, w, new_cols[0])
            else:
                variant = add_fn(features, prices, w, new_cols)
            m = eval_variant(variant, labels, cols, label_name)
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
    rdf.to_csv("data/test_indicator_windows_results.csv", index=False)
    logger.info("Saved data/test_indicator_windows_results.csv")

    winners = {g: w for g, w in best_per_group.items() if w is not None}
    if not winners:
        logger.info("=" * 100)
        logger.info("No group beat baseline individually -- current indicator windows already hold up, nothing to combine.")
        logger.info("=" * 100)
        return

    logger.info("=" * 100)
    logger.info("Best candidates (to combine and re-check): %s", winners)
    logger.info("=" * 100)
    combined = features
    drop_cols = []
    for group, w in winners.items():
        candidates, cols, add_fn = GROUPS[group]
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
                     "Current indicator windows kept unchanged.")


if __name__ == "__main__":
    run()
