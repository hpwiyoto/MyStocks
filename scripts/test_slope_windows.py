"""Are the slope window sizes actually optimal, or just conventional
defaults that were never empirically swept -- direct user request:
"apakah bisa dicek untuk semua parameter yang menggunakan slope...
apakah penentuan slopenya sudah paling optimal".

features/technical.py's `_slope(series, n) = (series - series.shift(n))
/ n` is used on 9 different series with 9 hand-picked windows (3d, 5d, or
10d depending on the column) -- picked by convention when v1-v5 were
built, never swept against alternatives. Only 4 of those slope columns
are actual SWING MODEL FEATURES though (the rest are excluded from
training entirely, either as ABSOLUTE_SCALE_COLS -- ema20_slope_5d,
sma50_slope_10d, volume_slope_5d, obv_slope_5d -- or not merged into the
panel at all -- macd_hist_slope_3d): rsi_slope_3d, rsi_slope_5d,
cmf_slope_5d, mfi_slope_5d. This script only covers those 4 -- a wrong
window on a column the model never sees can't be hurting it.

Each candidate is recomputed directly from the ALREADY-COMPUTED rsi_14/
cmf_20/mfi_14 columns using features/technical.py's own _slope() formula
-- no need to touch raw price data, the underlying indicator values are
already in the parquet export.

Methodology: one indicator at a time, holding everything else at the
CURRENT shipped 44-feature set (target=10%/stop=5%/horizon=5d, the
config search_swing_target.py adopted). RSI's baseline already uses TWO
windows (3d AND 5d, both kept as separate features) -- its W-variants
collapse this to ONE column, so a baseline win there specifically tests
"is keeping both windows worth it", not just "which single window is
best". CMF/MFI's baseline uses one window (5d) -- straightforward
swap-and-compare. Same pooled (trade-weighted) walk-forward methodology
as every other feature test here, threshold held FIXED at the shipped
0.60 (this tests feature REPRESENTATION, not threshold re-tuning --
re-tuning per variant would confound the comparison).

RESULT (2026-09-14): current windows NOT beaten -- no change adopted.
Baseline (current 3d+5d RSI / 5d CMF / 5d MFI): pooled n=762 wilson_lb=
68.62%. Individually, single-window sweeps found apparent winners: RSI
collapsed to ONE window at 7d (n=775, wilson_lb=69.80%, +1.18pp), CMF at
10d (n=792, wilson_lb=70.04%, +1.42pp), MFI at 7d (n=792, wilson_lb=
69.13%, +0.29pp -- much smaller, closer to noise-level than the other
two). Combining all three apparent "winners" together and re-testing
gave wilson_lb=67.40% (n=836), WORSE than the untouched baseline by
-1.22pp -- the individual gains did NOT stack, they reversed entirely.
This is the classic one-variable-at-a-time sweep pitfall this project
explicitly tries to avoid elsewhere (see e.g. features/regime.py's
grid-search-overfitting note): testing several independent variants and
picking whichever looks best is prone to surfacing sampling noise
(n~800 pooled trades total, each variant subtly shifts which rows cross
BUY_THRESHOLD=0.60) as if it were a real, exploitable pattern.
Conclusion: the CURRENT slope windows (RSI 3d+5d dual, CMF 5d, MFI 5d)
hold up fine against realistic alternatives once checked properly (i.e.
combined, not cherry-picked one at a time) -- not proven to be the
mathematically optimal windows in some absolute sense (that's not a
well-posed question for a walk-forward-noisy metric like this), but not
beaten by anything found here either, so left unchanged.

Usage:
    python -m scripts.test_slope_windows
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

logger = get_logger("scripts.test_slope_windows")

WINDOWS = [2, 3, 5, 7, 10, 15]
# (group name, base indicator column, current slope column(s) to remove for a variant)
INDICATOR_GROUPS = {
    "rsi": ("rsi_14", ["rsi_slope_3d", "rsi_slope_5d"]),
    "cmf": ("cmf_20", ["cmf_slope_5d"]),
    "mfi": ("mfi_14", ["mfi_slope_5d"]),
}


def add_custom_slope(features: pd.DataFrame, base_col: str, window: int, new_col: str) -> pd.DataFrame:
    features = features.copy()
    grouped = features.groupby("stock_code")[base_col]
    features[new_col] = (features[base_col] - grouped.shift(window)) / window
    return features


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


def evaluate(df, feature_cols, splits, label_name):
    y, prob = run_walk_forward_pooled(df, feature_cols, splits, SWING_XGB_PARAMS, label_name)
    roc_auc = ml_metrics(y, prob, 0.5)["roc_auc"] if len(y) else float("nan")
    buy_mask = prob >= BUY_THRESHOLD
    n = int(buy_mask.sum())
    wins = int(y[buy_mask].sum()) if n else 0
    precision = wins / n if n else float("nan")
    lb = wilson_lower_bound(wins, n) if n else 0.0
    return {"roc_auc": roc_auc, "n": n, "wins": wins, "precision": precision, "wilson_lb": lb}


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
    df, feature_cols = prepare_panel(features, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    baseline_metrics = evaluate(df, feature_cols, splits, "baseline")
    logger.info("[baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "window": "current", **baseline_metrics})

    for group, (base_col, current_cols) in INDICATOR_GROUPS.items():
        logger.info("#" * 90)
        logger.info("GROUP: %s (base column: %s, replacing: %s)", group, base_col, current_cols)
        logger.info("#" * 90)
        for w in WINDOWS:
            label_name = f"{group}_w{w}"
            new_col = f"{group}_slope_custom_{w}d"
            variant_features = add_custom_slope(features, base_col, w, new_col)
            variant_features_for_panel = variant_features.drop(columns=[c for c in current_cols if c in variant_features.columns])
            vdf, vfeature_cols = prepare_panel(variant_features_for_panel, labels)
            # prepare_panel's feature_cols already excludes NON_FEATURE_COLS/
            # ABSOLUTE_SCALE_COLS -- new_col isn't in ABSOLUTE_SCALE_COLS so
            # it's included automatically; current_cols are already dropped
            # from the source dataframe above so they can't reappear.
            vdates = vdf["date"].to_numpy()
            vsplits = walk_forward_splits(vdates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
            m = evaluate(vdf, vfeature_cols, vsplits, label_name)
            logger.info("[%s] window=%dd roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                        group, w, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                        m["wilson_lb"] - baseline_metrics["wilson_lb"])
            results.append({"group": group, "window": w, **m})

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS (single-variable sweeps):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_slope_windows_results.csv", index=False)
    logger.info("Saved data/test_slope_windows_results.csv")

    # Sanity check, not optional: whichever window looked best PER GROUP in
    # the single-variable sweep above, combine them together and re-test --
    # picking a per-group winner without this step is exactly the kind of
    # one-variable-at-a-time overfitting this project tries to avoid
    # elsewhere (see the RESULT note above). If the combined variant doesn't
    # beat baseline too, the individual "wins" were sampling noise.
    best_by_group = (
        rdf[rdf["group"] != "baseline"]
        .loc[rdf[rdf["group"] != "baseline"].groupby("group")["wilson_lb"].idxmax()]
    )
    logger.info("=" * 100)
    logger.info("Best window per group (to combine and re-check): %s",
                dict(zip(best_by_group["group"], best_by_group["window"])))
    logger.info("=" * 100)

    combined_features = features.copy()
    dropped = []
    for _, row in best_by_group.iterrows():
        group, w = row["group"], int(row["window"])
        base_col, current_cols = INDICATOR_GROUPS[group]
        new_col = f"{group}_slope_best_{w}d"
        combined_features = add_custom_slope(combined_features, base_col, w, new_col)
        dropped.extend([c for c in current_cols if c in combined_features.columns])
    combined_features = combined_features.drop(columns=dropped)
    cdf, cfeature_cols = prepare_panel(combined_features, labels)
    cdates = cdf["date"].to_numpy()
    csplits = walk_forward_splits(cdates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    combined_metrics = evaluate(cdf, cfeature_cols, csplits, "combined_best")
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
