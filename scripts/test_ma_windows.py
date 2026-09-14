"""Are the moving-average choices behind the Swing model's 2 MA-derived
features optimal, or just conventional defaults never swept -- direct
user follow-up to scripts/test_slope_windows.py: "apakah mungkin juga MA
yang digunakan bisa jadi kombinasi MA saat ini masih belum optimal, dan
pemilihan MA juga belum pas".

features/technical.py's compute_trend() computes 6 moving averages (SMA
20/50/200, EMA 9/20/50) but only 2 DERIVED columns from them are actual
Swing model features (the raw SMA/EMA levels themselves are
ABSOLUTE_SCALE_COLS, excluded from training):
  - price_vs_sma50_pct: (close - SMA50) / SMA50 * 100 -- why SMA50 and
    not SMA20/100/150/200, and why a SIMPLE MA rather than an EMA?
  - ema9_vs_ema20_pct: (EMA9 - EMA20) / EMA20 * 100 -- a classic fast/
    slow crossover, but why THIS pair and not 5/20, 12/26 (MACD's own
    standard pair), 8/21, 5/13 (Fibonacci-adjacent pairs seen elsewhere
    in TA convention)?

Recomputed directly from CLOSE (data/export_for_colab_prices.parquet),
not from the features parquet -- unlike the slope test, there's no
already-computed "generic MA" column to reuse, every candidate period/
type needs its own fresh rolling/ewm pass.

Same methodology as test_slope_windows.py, including the same
combined-recheck safeguard that test found necessary: one group at a
time (holding the other 43 features at the shipped baseline), sweep
candidates via the same pooled walk-forward methodology (target=10%/
stop=5%/horizon=5d, threshold=0.60 held fixed), THEN combine whichever
candidate wins each group and re-test before drawing any conclusion --
a per-group winner that doesn't survive combination is noise, not a
real improvement (exactly what happened with the slope windows).

RESULT (2026-09-14): current MA choices NOT beaten -- no change adopted.
Baseline (SMA50 for price-vs-MA, EMA9/20 crossover): pooled n=762
wilson_lb=68.62%. Individually, price_vs_sma20 looked best in Group A
(n=799, wilson_lb=69.27%, +0.65pp) and ema5_vs_ema20 looked best in
Group B (n=790, wilson_lb=69.32%, +0.69pp) -- both modest but real-
looking gains over the current SMA50/EMA9-20 defaults. Longer MAs
(SMA100/150/200, EMA50) were all flat-to-worse, so this isn't "any MA
beats the current one" -- shorter/faster MAs specifically looked better
in isolation. But combining both apparent winners (price_vs_sma20 +
ema5_vs_ema20) and re-testing gave wilson_lb=68.29% (n=817), WORSE than
the untouched baseline by -0.33pp -- same exact pattern found in
scripts/test_slope_windows.py: individually-promising single-variable
wins did not stack, they reversed once combined. Conclusion: the CURRENT
MA choices (SMA50 for price-vs-MA, EMA9/20 crossover) hold up fine
against realistic alternatives once checked combined rather than
cherry-picked one at a time -- left unchanged.

Usage:
    python -m scripts.test_ma_windows
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

logger = get_logger("scripts.test_ma_windows")

# Group A: replaces price_vs_sma50_pct -- (ma_type, period), "sma" or "ema"
PRICE_VS_MA_CANDIDATES = [
    ("sma", 10), ("sma", 20), ("sma", 100), ("sma", 150), ("sma", 200), ("ema", 50),
]
# Group B: replaces ema9_vs_ema20_pct -- (fast, slow) EMA pairs
CROSSOVER_CANDIDATES = [
    (5, 20), (10, 30), (12, 26), (8, 21), (5, 13),
]


def compute_ma(close_by_ticker: pd.Series, stock_code: pd.Series, ma_type: str, period: int) -> pd.Series:
    grouped = close_by_ticker.groupby(stock_code)
    if ma_type == "sma":
        return grouped.transform(lambda s: s.rolling(period, min_periods=period).mean())
    return grouped.transform(lambda s: s.ewm(span=period, adjust=False).mean())


def add_price_vs_ma(features: pd.DataFrame, prices: pd.DataFrame, ma_type: str, period: int, new_col: str) -> pd.DataFrame:
    # Sorted by stock_code+date BEFORE any rolling/ewm computation -- a
    # rolling window over out-of-order rows within a ticker would be
    # silently wrong, and neither the raw parquet export nor a merge is
    # guaranteed to already be in that order (same convention as
    # features/wyckoff.py's compute_wyckoff_features).
    merged = features.merge(prices[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
    merged = merged.sort_values(["stock_code", "date"]).reset_index(drop=True)
    ma = compute_ma(merged["close"], merged["stock_code"], ma_type, period)
    merged[new_col] = (merged["close"] - ma) / ma * 100
    return merged.drop(columns=["close"])


def add_ema_crossover(features: pd.DataFrame, prices: pd.DataFrame, fast: int, slow: int, new_col: str) -> pd.DataFrame:
    merged = features.merge(prices[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
    merged = merged.sort_values(["stock_code", "date"]).reset_index(drop=True)
    ema_fast = compute_ma(merged["close"], merged["stock_code"], "ema", fast)
    ema_slow = compute_ma(merged["close"], merged["stock_code"], "ema", slow)
    merged[new_col] = (ema_fast - ema_slow) / ema_slow * 100
    return merged.drop(columns=["close"])


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


def eval_variant(features_variant, labels, drop_cols, label_name):
    panel_input = features_variant.drop(columns=[c for c in drop_cols if c in features_variant.columns])
    df, feature_cols = prepare_panel(panel_input, labels)
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    return evaluate(df, feature_cols, splits, label_name), df, feature_cols


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
    baseline_metrics, _, _ = eval_variant(features, labels, [], "baseline")
    logger.info("[baseline] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f",
                baseline_metrics["roc_auc"], baseline_metrics["n"], baseline_metrics["wins"],
                baseline_metrics["precision"], baseline_metrics["wilson_lb"])
    results.append({"group": "baseline", "candidate": "current", **baseline_metrics})

    logger.info("#" * 90)
    logger.info("GROUP A: price_vs_MA (replacing price_vs_sma50_pct)")
    logger.info("#" * 90)
    best_a = {"wilson_lb": baseline_metrics["wilson_lb"], "candidate": None}
    for ma_type, period in PRICE_VS_MA_CANDIDATES:
        label_name = f"price_vs_{ma_type}{period}"
        variant = add_price_vs_ma(features, prices, ma_type, period, "price_vs_ma_custom")
        m, _, _ = eval_variant(variant, labels, ["price_vs_sma50_pct"], label_name)
        logger.info("[%s] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                    label_name, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                    m["wilson_lb"] - baseline_metrics["wilson_lb"])
        results.append({"group": "price_vs_ma", "candidate": f"{ma_type}{period}", **m})
        if m["wilson_lb"] > best_a["wilson_lb"]:
            best_a = {"wilson_lb": m["wilson_lb"], "candidate": (ma_type, period)}

    logger.info("#" * 90)
    logger.info("GROUP B: EMA crossover (replacing ema9_vs_ema20_pct)")
    logger.info("#" * 90)
    best_b = {"wilson_lb": baseline_metrics["wilson_lb"], "candidate": None}
    for fast, slow in CROSSOVER_CANDIDATES:
        label_name = f"ema{fast}_vs_ema{slow}"
        variant = add_ema_crossover(features, prices, fast, slow, "ema_crossover_custom")
        m, _, _ = eval_variant(variant, labels, ["ema9_vs_ema20_pct"], label_name)
        logger.info("[%s] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                    label_name, m["roc_auc"], m["n"], m["wins"], m["precision"], m["wilson_lb"],
                    m["wilson_lb"] - baseline_metrics["wilson_lb"])
        results.append({"group": "ema_crossover", "candidate": f"{fast}/{slow}", **m})
        if m["wilson_lb"] > best_b["wilson_lb"]:
            best_b = {"wilson_lb": m["wilson_lb"], "candidate": (fast, slow)}

    rdf = pd.DataFrame(results)
    logger.info("=" * 100)
    logger.info("FULL RESULTS (single-variable sweeps):")
    logger.info("=" * 100)
    logger.info("\n%s", rdf.to_string(index=False))
    rdf.to_csv("data/test_ma_windows_results.csv", index=False)
    logger.info("Saved data/test_ma_windows_results.csv")

    # Sanity check, not optional -- same lesson as scripts/test_slope_
    # windows.py: a per-group winner that doesn't survive being combined
    # with the OTHER group's winner is noise, not a real improvement.
    if best_a["candidate"] is None and best_b["candidate"] is None:
        logger.info("=" * 100)
        logger.info("Neither group beat baseline individually -- current MA choices already hold up, nothing to combine.")
        logger.info("=" * 100)
        return

    logger.info("=" * 100)
    logger.info("Best candidates (to combine and re-check): group A=%s, group B=%s", best_a["candidate"], best_b["candidate"])
    logger.info("=" * 100)
    combined = features
    drop_cols = []
    if best_a["candidate"] is not None:
        ma_type, period = best_a["candidate"]
        combined = add_price_vs_ma(combined, prices, ma_type, period, "price_vs_ma_best")
        drop_cols.append("price_vs_sma50_pct")
    if best_b["candidate"] is not None:
        fast, slow = best_b["candidate"]
        combined = add_ema_crossover(combined, prices, fast, slow, "ema_crossover_best")
        drop_cols.append("ema9_vs_ema20_pct")
    combined_metrics, _, _ = eval_variant(combined, labels, drop_cols, "combined_best")
    logger.info("[combined_best] roc_auc=%.4f n=%d wins=%d precision=%.4f wilson_lb=%.4f (vs baseline %+.4f)",
                combined_metrics["roc_auc"], combined_metrics["n"], combined_metrics["wins"],
                combined_metrics["precision"], combined_metrics["wilson_lb"],
                combined_metrics["wilson_lb"] - baseline_metrics["wilson_lb"])
    if combined_metrics["wilson_lb"] > baseline_metrics["wilson_lb"]:
        logger.info("Combined variant BEATS baseline -- worth considering for adoption.")
    else:
        logger.info("Combined variant does NOT beat baseline -- individual single-variable "
                     "'wins' above were sampling noise, not a real combined improvement. "
                     "Current MA choices kept unchanged.")


if __name__ == "__main__":
    run()
