"""Backtest for features.expert_rules' "Golden Cross Awal" combination --
direct user follow-up to shipping the live version on Momentum Screener:
find out whether it actually beats doing nothing, same "test before
trusting" posture as everything else in this project.

Same target/methodology as scripts/search_momentum_rules.py (+5%/-2.5%/
10 trading days triple barrier, no-lookahead as-of replay) for direct
comparability against the numbers already established there and in
scripts/grid_search_momentum_rules.py. As-of dates selected via
scripts.search_momentum_rules.select_as_of_dates (calendar-anchored,
not the old array-position stride -- see that function's docstring for
why the switch: the old method turned out to silently reshuffle which
dates get tested whenever unrelated historical data gets corrected,
which is exactly what happened partway through this feature's own
development, 2026-09-25).

NOTE: the v1/v2 backtest numbers referenced in features/expert_rules.py's
docstring were computed BEFORE this fix, using the old array-position
stride -- they are a real historical record of what those runs showed at
the time, but are not exactly reproducible with this version of the
script. Re-run this script for the current, methodology-fixed numbers.

Calls features.expert_rules.evaluate_expert_signal DIRECTLY on each
historical as-of-date's trailing window, rather than re-implementing the
same rules here -- this backtest tests what actually ships on the
Momentum Screener page, not a close approximation of it.

NOTE on an earlier mix-up (corrected here): the "Golden Cross" already
tested once in scripts/test_strategy_6_criteria.py (26.2% win rate,
below the 30.55% null baseline) was MACD-line-crosses-signal-line-below-
zero -- a DIFFERENT pattern some traders also call "golden cross", not
the price-moving-average (SMA50xSMA200 / SMA20xSMA50) crossover this
module and the user's rule actually mean. That prior result is NOT
directly informative about this one; this script is the first real test
of the price-MA version.

Usage:
    python -m scripts.backtest_expert_golden_cross
"""
import datetime as dt

import pandas as pd

from features.expert_rules import evaluate_expert_signal
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import select_as_of_dates, wilson_lower_bound

logger = get_logger("scripts.backtest_expert_golden_cross")

TARGET_PCT = 0.05
STOP_PCT = 0.025
HORIZON = 10
LOOKBACK_DAYS = 60   # matches app.data.load_screener_raw_panel's default window fed to the live rules
# WARMUP/stride selection delegated to scripts.search_momentum_rules.
# select_as_of_dates -- sma_200 not being ready yet is already handled by
# the per-row NaN check below (build_dataset), same as before; no need
# for a separate larger warmup just for that.


def triple_barrier_outcome(fwd: pd.DataFrame, entry_price: float) -> int | None:
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
    logger.info("Loading full price_history + feature_daily history...")
    df = pd.read_sql(
        """
        SELECT ph.stock_code, ph.date, ph.close, ph.high, ph.low, ph.volume,
               fd.rsi_14, fd.macd_hist, fd.macd_hist_slope_3d, fd.rvol_20,
               fd.sma_20, fd.sma_50, fd.sma_200,
               fd.distance_to_support_pct, fd.net_foreign_flow
        FROM price_history ph
        JOIN feature_daily fd ON fd.stock_code = ph.stock_code AND fd.date = ph.date
        WHERE ph.source_provider = 'yfinance'
        ORDER BY ph.stock_code, ph.date
        """,
        engine,
    )
    logger.info("Loaded %d rows across %d tickers", len(df), df["stock_code"].nunique())
    return df


def build_dataset() -> pd.DataFrame:
    """One row per (as_of_date, stock_code) that had enough history to
    evaluate the live expert-rule function AND a resolved forward
    outcome -- everything a candidate mask below could need, computed
    once via the SAME evaluate_expert_signal the live screener calls."""
    panel = load_full_panel()
    ticker_frames = {code: g.reset_index(drop=True) for code, g in panel.groupby("stock_code")}
    idx_by_date = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates_dt = sorted(dt.date.fromisoformat(d) for d in panel["date"].unique())
    as_of_dates = [d.isoformat() for d in select_as_of_dates(all_dates_dt)]
    logger.info("%d as-of dates", len(as_of_dates))

    rows = []
    for n_done, as_of in enumerate(as_of_dates):
        for code, g in ticker_frames.items():
            idx = idx_by_date[code].get(as_of)
            if idx is None or idx < 30:
                continue
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window = g.iloc[start:idx + 1]
            latest = window.iloc[-1]
            if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]) or pd.isna(latest["sma_200"]):
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue
            result = evaluate_expert_signal(window)
            rows.append({
                "as_of": as_of, "stock_code": code, "outcome": outcome,
                "passed": result["passed"], "golden_cross": result["golden_cross"],
                "cross_type": result["cross_type"], "macd_ok": result["macd_ok"],
                "volume_ok": result["volume_ok"], "rsi_ok": result["rsi_ok"],
                "foreign_flow_bonus": result["foreign_flow_bonus"],
                "near_support_bonus": result["near_support_bonus"], "score": result["score"],
            })
        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rows so far)", n_done + 1, len(as_of_dates), len(rows))

    out = pd.DataFrame(rows)
    logger.info("Dataset ready: %d resolved (ticker, as-of-date) rows", len(out))
    return out


def evaluate(df: pd.DataFrame, mask: pd.Series, label: str, null_rate: float) -> dict:
    sub = df[mask]
    n = len(sub)
    wins = int(sub["outcome"].sum())
    wr = wins / n if n else float("nan")
    return {
        "label": label, "n": n, "win_rate": wr,
        "lift_vs_null": wr / null_rate if n and null_rate else float("nan"),
        "wilson_lb": wilson_lower_bound(wins, n) if n else 0.0,
    }


def run():
    df = build_dataset()
    if df.empty:
        logger.error("Dataset is empty -- nothing to evaluate.")
        return
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE (every resolved ticker-date, zero filtering): n=%d win_rate=%.4f",
                len(df), null_rate)
    logger.info("=" * 70)

    candidates = [
        ("Golden Cross (any type) ALONE", df["golden_cross"]),
        ("Golden Cross MA50xMA200 ALONE", df["cross_type"] == "MA50xMA200"),
        ("Golden Cross MA20xMA50 ALONE", df["cross_type"] == "MA20xMA50"),
        ("MACD early-stage ALONE", df["macd_ok"]),
        ("Volume confirmation (RVOL>=1) ALONE", df["volume_ok"]),
        ("RSI normal (40-70) ALONE", df["rsi_ok"]),
        ("FULL COMBO (all 4 gates -- what ships live)", df["passed"]),
        ("FULL COMBO, MA50xMA200 only", df["passed"] & (df["cross_type"] == "MA50xMA200")),
        ("FULL COMBO, MA20xMA50 only", df["passed"] & (df["cross_type"] == "MA20xMA50")),
        ("FULL COMBO + foreign flow bonus", df["passed"] & df["foreign_flow_bonus"]),
        ("FULL COMBO + near-support bonus", df["passed"] & df["near_support_bonus"]),
        ("FULL COMBO, score>=85 ('Kuat' tier only)", df["passed"] & (df["score"] >= 85)),
    ]

    results = [evaluate(df, mask, label, null_rate) for label, mask in candidates]
    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 55)
    logger.info("\n%s", results_df.to_string(index=False))

    logger.info("=" * 70)
    logger.info("Ranked by Wilson 95%% lower bound (conservative -- penalizes small n, "
                "not just the raw point estimate)")
    logger.info("=" * 70)


if __name__ == "__main__":
    run()
