"""Historical backtest for the Momentum Screener -- unlike Swing/Turnaround,
this screener is a fixed rule set (RSI/MACD/regime/divergence), never a
trained model, so it has no existing walk-forward validation to point to
and its live `predictions`-equivalent history is empty (it's computed
on-demand, nothing is ever persisted). This replays its exact logic
(features.momentum_screener) at many past as-of dates using only data
available up to that date, then checks what REALLY happened afterward
using actual price_history -- the same triple-barrier definition Swing
itself is graded on (+5% target / -2.5% stop / 10 trading days), so the
resulting win rate is directly comparable to Swing's own reported 84.47%
(scripts/train_v5.py's walk-forward result) even though this is a
completely different kind of test (rule replay, not a trained model).

No lookahead risk here the way there would be for re-scoring a trained
model on old dates (that model was FIT on the full historical panel
including those exact rows, so redoing its own historical predictions
retroactively would leak information the live model never had — hence
why we cite Swing/Turnaround's ALREADY-existing out-of-sample walk-forward
numbers instead of recomputing them here). A fixed, human-authored rule
never "learned" from the data, so replaying it against real historical
rows is legitimate.

Usage:
    python -m scripts.backtest_momentum_screener
"""
import numpy as np
import pandas as pd

from features.momentum_screener import (
    DEFAULT_REGIME_PRIORITY,
    REGIME_PRIORITY,
    classify_macd_status,
    detect_bullish_divergence,
)
from pipeline.db import get_engine
from pipeline.logging_config import get_logger

logger = get_logger("scripts.backtest_momentum_screener")

TARGET_PCT = 0.05
STOP_PCT = 0.025
HORIZON = 10          # trading days -- identical to Swing's own definition
LOOKBACK_DAYS = 60    # trailing window the live screener itself uses
AS_OF_STRIDE = 10      # sample every Nth trading day (keeps runtime sane)
WARMUP_DATES = 120    # skip the very start of history (feature warmup)
TOP_N = 5             # how many top-priority picks per as-of date to grade
RSI_RANGE = (20, 60)
MACD_OK = {"Bullish Crossover", "Bullish"}


def triple_barrier_outcome(fwd: pd.DataFrame, entry_price: float) -> int | None:
    """fwd: up to HORIZON rows of (high, low) strictly after entry, ascending
    date. 1 = target hit before stop, 0 = stop hit before/without target,
    None = neither within the horizon (unresolved -- excluded, not forced)."""
    target = entry_price * (1 + TARGET_PCT)
    stop = entry_price * (1 - STOP_PCT)
    for _, row in fwd.iterrows():
        if row["low"] <= stop:
            return 0
        if row["high"] >= target:
            return 1
    return None


def load_full_panel() -> pd.DataFrame:
    engine = get_engine()
    logger.info("Loading full price_history + feature_daily history (this may take a bit)...")
    df = pd.read_sql(
        """
        SELECT ph.stock_code, ph.date, ph.close, ph.high, ph.low, ph.volume,
               fd.rsi_14, fd.macd, fd.macd_signal, fd.macd_hist, fd.macd_hist_slope_3d,
               fd.cmf_20, fd.rvol_20, fd.regime
        FROM price_history ph
        JOIN feature_daily fd ON fd.stock_code = ph.stock_code AND fd.date = ph.date
        WHERE ph.source_provider = 'yfinance'
        ORDER BY ph.stock_code, ph.date
        """,
        engine,
    )
    logger.info("Loaded %d rows across %d tickers", len(df), df["stock_code"].nunique())
    return df


def screen_one_date(ticker_frames: dict, as_of_idx_by_ticker: dict, as_of_date) -> pd.DataFrame:
    """Reconstruct what the live screener would have shown on `as_of_date`,
    using only each ticker's rows up to and including that date."""
    rows = []
    for code, g in ticker_frames.items():
        idx = as_of_idx_by_ticker[code].get(as_of_date)
        if idx is None or idx < 30:  # not enough trailing history yet
            continue
        start = max(0, idx - LOOKBACK_DAYS + 1)
        window = g.iloc[start:idx + 1]
        latest = window.iloc[-1]
        if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]):
            continue
        div = detect_bullish_divergence(window)
        macd_status = classify_macd_status(window["macd_hist"])
        rows.append({
            "stock_code": code, "close": latest["close"], "rsi_14": latest["rsi_14"],
            "macd_status": macd_status, "cmf_20": latest["cmf_20"], "regime": latest["regime"],
            "row_idx": idx, **div,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["divergence_tier"] = np.select(
        [out["divergence_rsi"] & out["divergence_macd"], out["divergence_rsi"] | out["divergence_macd"]],
        [0, 1], default=2,
    )
    out["regime_priority"] = out["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
    out["rsi_pivot_distance"] = (out["rsi_14"] - 50).abs()
    return out


def run():
    panel = load_full_panel()
    ticker_frames = {code: g.reset_index(drop=True) for code, g in panel.groupby("stock_code")}
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates = sorted(panel["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates to backtest (every %d trading days)", len(as_of_dates), AS_OF_STRIDE)

    topn_outcomes, base_outcomes = [], []
    for n_done, as_of in enumerate(as_of_dates):
        screened = screen_one_date(ticker_frames, as_of_idx_by_ticker, as_of)
        if screened.empty:
            continue
        base = screened[
            screened["rsi_14"].between(*RSI_RANGE)
            & screened["macd_status"].isin(MACD_OK)
            & (screened["cmf_20"] > 0)
        ]
        if base.empty:
            continue
        ranked = base.sort_values(
            ["divergence_tier", "regime_priority", "rsi_pivot_distance", "divergence_age_days", "stock_code"],
            ascending=[True, True, True, True, True], na_position="last",
        )
        top = ranked.head(TOP_N)

        for _, r in base.iterrows():
            code, idx = r["stock_code"], int(r["row_idx"])
            g = ticker_frames[code]
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(r["close"]))
            if outcome is not None:
                base_outcomes.append(outcome)
                if code in set(top["stock_code"]):
                    topn_outcomes.append(outcome)

        if (n_done + 1) % 10 == 0:
            logger.info("... %d/%d as-of dates done (base n=%d, top-N n=%d so far)",
                        n_done + 1, len(as_of_dates), len(base_outcomes), len(topn_outcomes))

    logger.info("=" * 70)
    logger.info("MOMENTUM SCREENER BACKTEST RESULT (target=+%.1f%%, stop=-%.1f%%, horizon=%dd)",
                TARGET_PCT * 100, STOP_PCT * 100, HORIZON)
    logger.info("=" * 70)
    if base_outcomes:
        base_arr = np.array(base_outcomes)
        logger.info("BASE FILTER (RSI 20-60 + Bullish MACD + CMF>0, no ranking): n=%d win_rate=%.4f",
                    len(base_arr), base_arr.mean())
    if topn_outcomes:
        top_arr = np.array(topn_outcomes)
        logger.info("TOP-%d BY PRIORITY (divergence+regime+RSI ranking): n=%d win_rate=%.4f",
                    TOP_N, len(top_arr), top_arr.mean())
    else:
        logger.warning("No resolved top-N outcomes -- TOP_N or AS_OF_STRIDE may need adjusting")


if __name__ == "__main__":
    run()
