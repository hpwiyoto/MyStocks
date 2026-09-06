"""Systematic search for a Momentum Screener rule combination that actually
beats doing nothing -- follow-up to scripts/backtest_momentum_screener.py,
which found the CURRENT filter+ranking (RSI 20-60 + Bullish MACD + CMF>0,
then divergence/regime/RSI-pivot priority) indistinguishable from its own
unranked base rate (24.78% vs 23.17%, well within noise).

Same no-lookahead argument as that script (a fixed rule set never
"learned" from the data, so replaying it against real historical rows at
many past as-of dates carries no leakage risk) and the same grading target
as Swing (+5% / -2.5% / 10 trading days) for direct comparability.

Restructured for search efficiency: the expensive part (reconstructing
RSI/MACD/divergence/regime state AND the actual forward outcome for every
ticker at every as-of date) is done ONCE and cached; each candidate rule
combination is then just a cheap boolean mask + groupby over that cached
table, so dozens of combinations can be tried in seconds instead of
re-running the full screen+outcome pass per candidate.

Usage:
    python -m scripts.search_momentum_rules
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

logger = get_logger("scripts.search_momentum_rules")

TARGET_PCT = 0.05
STOP_PCT = 0.025
HORIZON = 10
LOOKBACK_DAYS = 60
AS_OF_STRIDE = 10
WARMUP_DATES = 120


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


def build_dataset() -> pd.DataFrame:
    """One row per (as_of_date, stock_code) with every indicator the
    screener computes PLUS the real triple-barrier outcome that followed --
    everything a candidate rule combination could need, computed once."""
    panel = load_full_panel()
    ticker_frames = {code: g.reset_index(drop=True) for code, g in panel.groupby("stock_code")}
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates = sorted(panel["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates", len(as_of_dates))

    rows = []
    for n_done, as_of in enumerate(as_of_dates):
        for code, g in ticker_frames.items():
            idx = as_of_idx_by_ticker[code].get(as_of)
            if idx is None or idx < 30:
                continue
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window = g.iloc[start:idx + 1]
            latest = window.iloc[-1]
            if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]):
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue
            div = detect_bullish_divergence(window)
            macd_status = classify_macd_status(window["macd_hist"])
            rows.append({
                "as_of": as_of, "stock_code": code, "close": latest["close"],
                "rsi_14": latest["rsi_14"], "macd_status": macd_status,
                "macd_hist_slope_3d": latest["macd_hist_slope_3d"],
                "cmf_20": latest["cmf_20"], "rvol_20": latest["rvol_20"], "regime": latest["regime"],
                "outcome": outcome, **div,
            })
        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rows so far)", n_done + 1, len(as_of_dates), len(rows))

    out = pd.DataFrame(rows)
    out["divergence_tier"] = np.select(
        [out["divergence_rsi"] & out["divergence_macd"], out["divergence_rsi"] | out["divergence_macd"]],
        [0, 1], default=2,
    )
    out["regime_priority"] = out["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
    out["rsi_pivot_distance"] = (out["rsi_14"] - 50).abs()
    logger.info("Dataset ready: %d resolved (ticker, as-of-date) rows", len(out))
    return out


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """95%% lower confidence bound on a win rate -- ranks candidates by
    how confident we can be the true rate beats the null, not just the
    raw point estimate (which is noisy at small n)."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)
    return (centre - adj) / denom


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
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE (every resolved ticker-date, zero filtering): n=%d win_rate=%.4f",
                len(df), null_rate)
    logger.info("=" * 70)

    candidates = []

    # --- Individual factors in isolation, to see which ones carry any
    # signal at all before combining them ---
    candidates.append(("RSI 20-60", df["rsi_14"].between(20, 60)))
    candidates.append(("RSI 30-50 (narrower, closer to the pivot)", df["rsi_14"].between(30, 50)))
    candidates.append(("RSI < 30 (oversold)", df["rsi_14"] < 30))
    candidates.append(("RSI 40-60", df["rsi_14"].between(40, 60)))
    candidates.append(("MACD Bullish/Bullish Crossover", df["macd_status"].isin(["Bullish", "Bullish Crossover"])))
    candidates.append(("MACD Bullish Crossover only (fresh)", df["macd_status"] == "Bullish Crossover"))
    candidates.append(("CMF > 0", df["cmf_20"] > 0))
    candidates.append(("CMF > 0.1 (stronger accumulation)", df["cmf_20"] > 0.1))
    candidates.append(("RVOL >= 1", df["rvol_20"] >= 1))
    candidates.append(("RVOL >= 1.5", df["rvol_20"] >= 1.5))
    candidates.append(("Momentum histogram menguat (slope>0)", df["macd_hist_slope_3d"] > 0))
    candidates.append(("Divergence Ganda only", df["divergence_tier"] == 0))
    candidates.append(("Divergence Ganda or Tunggal", df["divergence_tier"] < 2))
    candidates.append(("Regime = early_reversal", df["regime"] == "early_reversal"))
    candidates.append(("Regime = bullish", df["regime"] == "bullish"))
    candidates.append(("Regime in {early_reversal, bullish}", df["regime"].isin(["early_reversal", "bullish"])))
    candidates.append(("Regime = accumulation", df["regime"] == "accumulation"))
    candidates.append(("Regime = bottoming", df["regime"] == "bottoming"))
    candidates.append(("Regime = sideways", df["regime"] == "sideways"))

    # --- The shipped combination, for reference ---
    shipped = (
        df["rsi_14"].between(20, 60)
        & df["macd_status"].isin(["Bullish", "Bullish Crossover"])
        & (df["cmf_20"] > 0)
    )
    candidates.append(("SHIPPED: RSI 20-60 + Bullish MACD + CMF>0", shipped))

    # --- A few combined candidates worth testing directly ---
    candidates.append((
        "RSI 30-50 + MACD Bullish + CMF>0.1",
        df["rsi_14"].between(30, 50) & df["macd_status"].isin(["Bullish", "Bullish Crossover"]) & (df["cmf_20"] > 0.1),
    ))
    candidates.append((
        "Regime early_reversal/bullish + CMF>0",
        df["regime"].isin(["early_reversal", "bullish"]) & (df["cmf_20"] > 0),
    ))
    candidates.append((
        "Regime early_reversal/bullish + RVOL>=1.5",
        df["regime"].isin(["early_reversal", "bullish"]) & (df["rvol_20"] >= 1.5),
    ))
    candidates.append((
        "Regime early_reversal/bullish + CMF>0 + RVOL>=1.5",
        df["regime"].isin(["early_reversal", "bullish"]) & (df["cmf_20"] > 0) & (df["rvol_20"] >= 1.5),
    ))
    candidates.append((
        "Divergence Ganda + Regime early_reversal/bullish",
        (df["divergence_tier"] == 0) & df["regime"].isin(["early_reversal", "bullish"]),
    ))

    # --- Follow-ups on the first pass's biggest surprise: "bottoming" (a
    # BAD regime, ranked LOW by the shipped REGIME_PRIORITY) clearing the
    # null baseline more convincingly than anything ranked ABOVE it,
    # including early_reversal/bullish. Testing whether combining it with
    # other factors pushes the edge further, or whether it's already close
    # to as good as this rule-based approach gets. ---
    candidates.append(("Regime = bottoming + RSI 30-50", (df["regime"] == "bottoming") & df["rsi_14"].between(30, 50)))
    candidates.append(("Regime = bottoming + RSI < 40", (df["regime"] == "bottoming") & (df["rsi_14"] < 40)))
    candidates.append(("Regime = bottoming + CMF > 0", (df["regime"] == "bottoming") & (df["cmf_20"] > 0)))
    candidates.append(("Regime = bottoming + RVOL >= 1", (df["regime"] == "bottoming") & (df["rvol_20"] >= 1)))
    candidates.append(("Regime = bottoming + momentum menguat", (df["regime"] == "bottoming") & (df["macd_hist_slope_3d"] > 0)))
    candidates.append(("Regime = bottoming + MACD Bullish Crossover", (df["regime"] == "bottoming") & (df["macd_status"] == "Bullish Crossover")))
    candidates.append(("Regime in {bottoming, bearish}", df["regime"].isin(["bottoming", "bearish"])))
    candidates.append(("Regime = bearish", df["regime"] == "bearish"))
    candidates.append(("Regime = bottoming + Divergence any", (df["regime"] == "bottoming") & (df["divergence_tier"] < 2)))

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
