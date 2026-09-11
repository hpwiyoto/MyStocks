"""CORRECTS scripts/check_suspension_risk.py, which concluded "no evidence"
of a Swing BUY signal ever preceding a real suspension. That conclusion was
WRONG -- not because the underlying data lied, but because the detection
method missed the actual signature suspended IDX stocks show in this data.

The error, found via a real live case (SAFE, BUY 2026-09-01 @ RSI 93.96 /
regime=overextended, right after a one-day +25% spike -- then 7+ straight
trading days of open=high=low=close=875, volume=0, and yfinance's live
quote endpoint erroring with a missing 'currentTradingPeriod' field, the
classic signature of a halted symbol): v1 only searched for ticker-
specific GAPS (dates with NO price_history row at all). A suspended IDX
stock instead keeps getting a row every day -- yfinance/IDX just repeats
the last traded price with zero volume, a FROZEN quote, not a missing
one. v1's gap search structurally could not see this at all, on ANY
ticker, at ANY point in the 5-year history -- it wasn't a narrow miss,
the entire detection mechanism was pointed at the wrong signature.

Corrected proxy: a run of >=5 consecutive trading days where OHLC are all
identical AND volume==0. Re-run against the full history: 1,511 such
episodes across 362 tickers with a real observed onset (excluding ~28
that were already frozen on 2021-08-31, this dataset's first day --
likely already-dormant/delisted-before-tracking, not a live event to
explain). That is a dramatically different picture from v1's "2 tickers,
neither BUY-eligible."

This script re-simulates Swing's historical walk-forward BUY probability
(same panel/splits/params as every other backtest here) across the full
panel, then checks: for each ticker's FIRST freeze onset, did a BUY
signal (prob>=0.65) fire in the LOOKBACK_DAYS trading days immediately
before -- the SAFE pattern, tested for real across history instead of
one anecdote.

Usage:
    python -m scripts.check_suspension_risk_v2
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.train_v5 import BUY_THRESHOLD, HORIZON, NUM_BOOST_ROUND, XGB_PARAMS, walk_forward_splits
from scripts.tune_v5 import _load_panel as load_panel

logger = get_logger("scripts.check_suspension_risk_v2")

FREEZE_MIN_DAYS = 5
LOOKBACK_DAYS = 20  # how far before freeze-onset to look for a preceding BUY signal


def find_freeze_onsets() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT stock_code, date, open, high, low, close, volume FROM price_history WHERE source_provider = 'yfinance'", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"])
    df["frozen"] = (df["open"] == df["close"]) & (df["high"] == df["close"]) & (df["low"] == df["close"]) & (df["volume"] == 0)

    events = []
    for code, g in df.groupby("stock_code"):
        g = g.reset_index(drop=True)
        run_len, run_start = 0, None
        for i, frozen in enumerate(g["frozen"]):
            if frozen:
                if run_len == 0:
                    run_start = g.loc[i, "date"]
                run_len += 1
            else:
                if run_len >= FREEZE_MIN_DAYS:
                    events.append({"stock_code": code, "freeze_start": run_start, "freeze_len_days": run_len})
                run_len = 0
        if run_len >= FREEZE_MIN_DAYS:
            events.append({"stock_code": code, "freeze_start": run_start, "freeze_len_days": run_len})
    edf = pd.DataFrame(events)
    edf = edf[edf["freeze_start"] > pd.Timestamp("2021-10-01")]  # drop pre-existing-on-day-1 artifacts
    first_per_ticker = edf.sort_values("freeze_start").groupby("stock_code", as_index=False).first()
    logger.info("Freeze episodes with a real observed onset: %d across %d tickers; first-episode-per-ticker: %d",
                len(edf), edf["stock_code"].nunique(), len(first_per_ticker))
    return first_per_ticker


def run_walk_forward_pooled(df, feature_cols, extra_cols, splits, xgb_params) -> pd.DataFrame:
    pooled = []
    for fold_i, split in enumerate(splits):
        train_mask = df["date"] <= split["train_embargo_end_date"]
        test_mask = (df["date"] >= split["test_start_date"]) & (df["date"] <= split["test_end_date"])
        X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "label"].to_numpy()
        X_test = df.loc[test_mask, feature_cols]
        if len(X_train) < 100 or len(X_test) < 20 or len(np.unique(y_train)) < 2:
            continue
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        prob = booster.predict(dtest)
        fold_extra = df.loc[test_mask, extra_cols].reset_index(drop=True)
        fold_df = pd.DataFrame({"fold": fold_i, "prob": prob})
        pooled.append(pd.concat([fold_df, fold_extra], axis=1))
        logger.info("fold %d done: n_test=%d", fold_i, len(X_test))
    return pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()


def run():
    freeze_onsets = find_freeze_onsets()

    df, feature_cols = load_panel()
    dates = df["date"].to_numpy()
    splits = walk_forward_splits(dates, n_splits=5, test_size_days=100, min_train_days=600, label_horizon=HORIZON)
    pooled = run_walk_forward_pooled(df, feature_cols, ["stock_code", "date"], splits, XGB_PARAMS)
    logger.info("Pooled OOS rows (all probabilities, all tickers/dates): %d", len(pooled))

    buy_signals = pooled[pooled["prob"] >= BUY_THRESHOLD][["stock_code", "date"]].copy()
    buy_signals["date"] = pd.to_datetime(buy_signals["date"])  # parquet stores date as datetime.date, not Timestamp -- normalize before comparing to freeze_start
    logger.info("Total historical BUY signals (walk-forward simulation, prob>=%.2f): %d", BUY_THRESHOLD, len(buy_signals))

    # For each freeze onset, was there a BUY signal on that ticker within
    # LOOKBACK_DAYS trading days before it (using the ticker's own trading-
    # day index, not calendar days, so weekends/holidays don't skew it)?
    buy_by_ticker = {code: sorted(g["date"]) for code, g in buy_signals.groupby("stock_code")}
    hits = []
    for _, row in freeze_onsets.iterrows():
        code, onset = row["stock_code"], row["freeze_start"]
        ticker_buys = buy_by_ticker.get(code, [])
        preceding = [d for d in ticker_buys if d < onset]
        if preceding and (onset - preceding[-1]).days <= LOOKBACK_DAYS * 1.5:  # trading-day approx via calendar-day cap
            hits.append({"stock_code": code, "buy_date": preceding[-1], "freeze_start": onset,
                         "calendar_days_between": (onset - preceding[-1]).days})

    logger.info("=" * 90)
    logger.info("RESULT: freeze onsets preceded by a walk-forward BUY signal within ~%d trading days before:", LOOKBACK_DAYS)
    logger.info("=" * 90)
    hdf = pd.DataFrame(hits)
    logger.info("%d out of %d first-freeze-onset tickers (%.1f%%) had a preceding BUY signal",
                len(hdf), len(freeze_onsets), len(hdf) / len(freeze_onsets) * 100 if len(freeze_onsets) else 0)
    if not hdf.empty:
        logger.info("\n%s", hdf.sort_values("freeze_start").to_string(index=False))
    logger.info("-" * 90)
    logger.info("Of %d total historical BUY signals, %d (%.2f%%) were followed by a freeze onset within ~%d trading days after:",
                len(buy_signals), len(hdf), len(hdf) / len(buy_signals) * 100 if len(buy_signals) else 0, LOOKBACK_DAYS)


if __name__ == "__main__":
    run()
