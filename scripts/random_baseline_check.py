"""Sanity check requested directly: over the last N trading days (default
50) with a resolved 10-day-forward outcome, compare a RANDOM sample of
(ticker, date) picks against every pick the v5 model would have called BUY
across that same window -- using the ACTUAL subsequent price data (already
resolved by now) to see what really happened, not a simulation.

A single day is too sparse to compare on its own (v5 fires very few BUY
signals per day by design -- see chat), so this pools every model BUY
across the whole window, same idea as the walk-forward validation's pooled
fold predictions.

Usage:
    python -m scripts.random_baseline_check [n_days_back] [random_sample_size]
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from engine.decision import BUY_THRESHOLD, decide
from pipeline.logging_config import get_logger
from scripts.train_v5 import FEATURES_PATH, PRICES_PATH, STOP_PCT, TARGET_PCT, build_panel_labels

logger = get_logger("scripts.random_baseline_check")


def run(n_days_back: int = 50, random_sample_size: int = 100, seed: int = 42):
    features = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labels = build_panel_labels(prices)  # real forward-looking outcome, already resolved

    all_dates = np.sort(features["date"].unique())
    if n_days_back >= len(all_dates):
        raise ValueError(f"only {len(all_dates)} dates available, can't go back {n_days_back}")
    window_dates = all_dates[-n_days_back:]
    logger.info("Window: %d trading days, %s to %s", len(window_dates), window_dates[0], window_dates[-1])

    window_features = features[features["date"].isin(window_dates)].copy()
    window = window_features.merge(labels[["stock_code", "date", "label"]], on=["stock_code", "date"], how="inner")
    window = window[window["label"].notna()].copy()  # only (ticker, date) pairs with a resolved 10-day outcome
    window = window.merge(prices[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
    logger.info("%d (ticker, date) pairs have a resolved label in this window", len(window))

    booster, meta = _load_v5()
    feature_cols = meta["feature_cols"]
    base_rate = meta["base_rate"]

    window_model_ready = _prepare_for_model(window, feature_cols)
    dmat = xgb.DMatrix(window_model_ready[feature_cols])
    window["probability"] = booster.predict(dmat)
    window["model_buy"] = window["probability"] >= BUY_THRESHOLD

    # Same decide() the live app uses (BUY/WATCH/AVOID, gocap floor
    # included) -- not just the raw >=threshold split -- so this reports
    # the ACTUAL decision a user would have seen for every pair.
    window["decision"] = [
        decide(p, base_rate, c, TARGET_PCT, STOP_PCT)["decision"] if pd.notna(c) else None
        for p, c in zip(window["probability"], window["close"])
    ]

    random_sample = window.sample(n=min(random_sample_size, len(window)), random_state=seed)
    model_buys = window[window["model_buy"]]

    logger.info("=" * 70)
    logger.info("RANDOM SAMPLE (n=%d, seed=%d, pooled across %d days): win_rate=%.3f (%d/%d hit +5%% "
                "before -2.5%% in 10 days)", len(random_sample), seed, len(window_dates),
                random_sample["label"].mean(), int(random_sample["label"].sum()), len(random_sample))
    if len(model_buys) > 0:
        logger.info("MODEL BUY (threshold=%.2f, n=%d, pooled): win_rate=%.3f (%d/%d hit target)",
                    BUY_THRESHOLD, len(model_buys), model_buys["label"].mean(),
                    int(model_buys["label"].sum()), len(model_buys))
        by_date = model_buys.groupby("date")["stock_code"].apply(list)
        for d, codes in by_date.items():
            logger.info("  %s: %s", d, sorted(codes))
    else:
        logger.info("MODEL BUY (threshold=%.2f): 0 signals in this whole window", BUY_THRESHOLD)
    logger.info("Overall base rate this window (%d pairs): %.3f", len(window), window["label"].mean())
    logger.info("=" * 70)
    logger.info("THRESHOLD TRADE-OFF (this window, %d trading days): signal frequency vs precision", len(window_dates))
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        picks = window[window["probability"] >= t]
        n = len(picks)
        wr = picks["label"].mean() if n else float("nan")
        per_day = n / len(window_dates)
        logger.info("  threshold=%.2f: n_signals=%6d (~%.1f/hari)  win_rate=%.3f", t, n, per_day, wr)
    logger.info("=" * 70)
    logger.info("WIN RATE BY DECISION (what a user actually would have seen, gocap floor included)")
    for dec in ("BUY", "WATCH", "AVOID"):
        sub = window[window["decision"] == dec]
        if len(sub) == 0:
            logger.info("  %-6s n=0", dec)
            continue
        logger.info("  %-6s n=%6d  win_rate=%.3f (%d hit +5%%, i.e. would have been a FALSE %s if so)",
                    dec, len(sub), sub["label"].mean(), int(sub["label"].sum()),
                    "negative -- AVOID/WATCH but it still rose" if dec != "BUY" else "positive")
    logger.info("=" * 70)


def _load_v5():
    import json
    booster = xgb.Booster()
    booster.load_model("models/direction_xgboost_v5.json")
    with open("models/direction_xgboost_v5_metadata.json") as f:
        meta = json.load(f)
    return booster, meta


def _prepare_for_model(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["has_similar_pattern"] = (df["similar_pattern_count"].fillna(0) > 0).astype(int)
    df["historical_win_rate"] = df["historical_win_rate"].fillna(0.5)
    for c in ("higher_high_20d", "higher_low_20d", "lower_high_20d", "lower_low_20d"):
        df[c] = df[c].astype(float)
    regime_dummies = pd.get_dummies(df["regime"], prefix="regime", dtype=float)
    df = pd.concat([df, regime_dummies], axis=1)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


if __name__ == "__main__":
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run(n_days, n_sample)
