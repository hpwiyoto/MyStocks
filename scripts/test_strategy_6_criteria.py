"""Backtests the user's proposed 6-criteria momentum strategy against the
SAME no-lookahead 5-year historical dataset used to validate the currently
shipped Momentum Screener rule (scripts/search_momentum_rules.py), then
searches for the best combination between the two rule sets.

The 6 criteria, as specified by the user (paraphrased):
1. Struktur Lantai: close within 2% above the rolling 50-day low.
2. Kondisi Jenuh Jual: RSI < 45.
3. Akumulasi Senyap: CMF < 0, but its 2-day slope is rising toward 0.
4. Bahan Bakar Gerakan: RVOL > 1 (volume above its own 20-day average).
5. Validasi Modal Bandar: close >= an Anchored VWAP, anchored at the date
   the rolling 50-day low occurred.
6. Pemicu Eksekusi Akhir: MACD line crosses above the signal line while
   still below the zero line ("golden cross below zero").

None of these existed in features/momentum_screener.py before this
script -- criterion 5 (Anchored VWAP) is a genuinely new indicator, and
1/2/4/6 are more precise versions of ideas the shipped rule only
approximates loosely (regime="bottoming" instead of "near the 50-day
low"; no RSI gate at all; RVOL>=0.8 instead of >1; MACD histogram
crossover without checking the zero-line position). Criterion 3 is a
close cousin of cmf_slope_5d, already tested in
scripts/search_momentum_rules.py and found to HURT (see that script's
"SHIPPED v2 + CMF slope>0" result) -- tested again here at a 2-day
window specifically, since the user's proposal uses a different lag,
not assumed to fail just because a close relative failed.

RESULT (2026-09-08): the FULL 6-criteria combo, ANDed together exactly as
specified, is unusable -- n=12, win_rate=25.0% (Wilson LB 8.9%), WORSE than
the 30.55% null baseline, and small enough that the point estimate itself
isn't trustworthy. Tested individually, 5 of the 6 criteria are weak-to-
harmful on their own (Struktur Lantai 33.6%, Jenuh Jual 32.6%, Akumulasi
Senyap 31.7%, Bahan Bakar 31.3%, Modal Bandar/AVWAP alone 29.1%, Golden
Cross below zero 26.2% -- all at or below the null baseline). But ONE
piece was genuinely additive when layered ON TOP of the already-shipped
rule: requiring close >= Anchored VWAP from the 50-day low, in addition
to the shipped bottoming+momentum+CMF<0+RVOL>=0.8 gate, improved n=958
win_rate=39.77% (Wilson LB 36.72%) to n=523 win_rate=42.45% (Wilson LB
38.28%) -- a real gain on the LB (the metric that penalizes a smaller n),
not just a smaller sample with a better point estimate. Every other
combination tried (adding Struktur Lantai, RSI<45, or both on top) scored
worse than this one. ADOPTED: features/momentum_screener.py's
is_validated_signal() now requires close_above_avwap, and
app/data.py/compute_screener_panel() compute the new AVWAP columns from
price_history.high/low. See features/momentum_screener.py's
VALIDATED_RVOL_THRESHOLD / AVWAP_WINDOW comments for the shipped version
of this finding.

Usage:
    python -m scripts.test_strategy_6_criteria
"""
import numpy as np
import pandas as pd

from features.momentum_screener import (
    DEFAULT_REGIME_PRIORITY,
    REGIME_PRIORITY,
    classify_macd_status,
    detect_bullish_divergence,
)
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import (
    AS_OF_STRIDE,
    HORIZON,
    LOOKBACK_DAYS,
    WARMUP_DATES,
    load_full_panel,
    triple_barrier_outcome,
    wilson_lower_bound,
)

logger = get_logger("scripts.test_strategy_6_criteria")

AVWAP_WINDOW = 50


def _add_new_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """g: one ticker's rows, ascending by date. Adds dist_to_low50_pct,
    avwap_from_low50, close_above_avwap, cmf_slope_2d, and
    golden_cross_below_zero -- computed once per ticker over its FULL
    history (not per as-of date), since each is a rolling/lag feature
    that only needs the past, same no-lookahead guarantee as every other
    column already in this panel."""
    close = g["close"].to_numpy(dtype=float)
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    volume = g["volume"].to_numpy(dtype=float)
    typical = (high + low + close) / 3
    n = len(close)

    dist_pct = np.full(n, np.nan)
    avwap = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - AVWAP_WINDOW + 1)
        seg_close = close[start:i + 1]
        anchor = start + int(np.argmin(seg_close))
        low_val = close[anchor]
        if low_val:
            dist_pct[i] = (close[i] - low_val) / low_val * 100
        seg_vol = volume[anchor:i + 1]
        vol_sum = seg_vol.sum()
        if vol_sum > 0:
            avwap[i] = (typical[anchor:i + 1] * seg_vol).sum() / vol_sum

    out = g.copy()
    out["dist_to_low50_pct"] = dist_pct
    out["avwap_from_low50"] = avwap
    out["close_above_avwap"] = close >= avwap

    macd, macd_signal = g["macd"].to_numpy(dtype=float), g["macd_signal"].to_numpy(dtype=float)
    crossed_up = np.concatenate([[False], (macd[1:] > macd_signal[1:]) & (macd[:-1] <= macd_signal[:-1])])
    out["golden_cross_below_zero"] = crossed_up & (macd < 0) & (macd_signal < 0)
    out["cmf_slope_2d"] = g["cmf_20"] - g["cmf_20"].shift(2)
    return out


def build_dataset() -> pd.DataFrame:
    panel = load_full_panel()
    ticker_frames = {}
    for code, g in panel.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        ticker_frames[code] = _add_new_indicators(g)
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
                "cmf_20": latest["cmf_20"], "cmf_slope_5d": latest["cmf_slope_5d"],
                "cmf_slope_2d": latest["cmf_slope_2d"],
                "rvol_20": latest["rvol_20"], "regime": latest["regime"],
                "dist_to_low50_pct": latest["dist_to_low50_pct"],
                "close_above_avwap": latest["close_above_avwap"],
                "golden_cross_below_zero": latest["golden_cross_below_zero"],
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
    logger.info("Dataset ready: %d resolved (ticker, as-of-date) rows", len(out))
    return out


def evaluate(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df[mask]
    n = len(sub)
    wins = int(sub["outcome"].sum())
    wr = wins / n if n else float("nan")
    return {"label": label, "n": n, "win_rate": wr, "wilson_lb": wilson_lower_bound(wins, n) if n else 0.0}


def run():
    df = build_dataset()
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(df), null_rate)
    logger.info("=" * 70)

    candidates = []

    # --- The 6 criteria individually, to see which (if any) carry signal alone ---
    candidates.append(("1. Struktur Lantai (dist to 50d-low <= 2%)", df["dist_to_low50_pct"] <= 2))
    candidates.append(("2. Jenuh Jual (RSI < 45)", df["rsi_14"] < 45))
    candidates.append(("3. Akumulasi Senyap (CMF<0 & slope_2d>0)", (df["cmf_20"] < 0) & (df["cmf_slope_2d"] > 0)))
    candidates.append(("4. Bahan Bakar (RVOL > 1)", df["rvol_20"] > 1))
    candidates.append(("5. Modal Bandar (close >= AVWAP from 50d-low)", df["close_above_avwap"]))
    candidates.append(("6. Golden Cross below zero", df["golden_cross_below_zero"]))

    # --- The full 6-criteria combo, exactly as specified ---
    full_combo = (
        (df["dist_to_low50_pct"] <= 2)
        & (df["rsi_14"] < 45)
        & (df["cmf_20"] < 0) & (df["cmf_slope_2d"] > 0)
        & (df["rvol_20"] > 1)
        & df["close_above_avwap"]
        & df["golden_cross_below_zero"]
    )
    candidates.append(("FULL 6-criteria combo (as specified)", full_combo))
    # Relaxed variant dropping the rarest/most restrictive single trigger
    # (golden_cross_below_zero, a single-day event) to see if the other 5
    # alone (a "setup" without requiring the exact trigger day) hold up
    # better at a usable sample size.
    setup_5_no_trigger = (
        (df["dist_to_low50_pct"] <= 2)
        & (df["rsi_14"] < 45)
        & (df["cmf_20"] < 0) & (df["cmf_slope_2d"] > 0)
        & (df["rvol_20"] > 1)
        & df["close_above_avwap"]
    )
    candidates.append(("5-of-6 (drop golden-cross trigger)", setup_5_no_trigger))

    # --- Currently shipped rule, for reference ---
    shipped = (
        (df["regime"] == "bottoming")
        & (df["macd_hist_slope_3d"] > 0)
        & (df["cmf_20"] < 0)
        & (df["rvol_20"] >= 0.8)
    )
    candidates.append(("SHIPPED: bottoming+momentum+CMF<0+RVOL>=0.8", shipped))

    # --- Hybrids: mix pieces from both rule sets to look for something
    # better than either alone. ---
    candidates.append(("Shipped + Struktur Lantai (<=2%)", shipped & (df["dist_to_low50_pct"] <= 2)))
    candidates.append(("Shipped + close>=AVWAP", shipped & df["close_above_avwap"]))
    candidates.append(("Shipped + RSI<45", shipped & (df["rsi_14"] < 45)))
    candidates.append(("Shipped + Struktur Lantai + AVWAP", shipped & (df["dist_to_low50_pct"] <= 2) & df["close_above_avwap"]))
    candidates.append(("Struktur Lantai + AVWAP + shipped's momentum+CMF<0", (df["dist_to_low50_pct"] <= 2) & df["close_above_avwap"] & (df["macd_hist_slope_3d"] > 0) & (df["cmf_20"] < 0)))
    candidates.append(("Struktur Lantai + AVWAP + RVOL>=0.8 (shipped's looser volume bar)", (df["dist_to_low50_pct"] <= 2) & df["close_above_avwap"] & (df["rvol_20"] >= 0.8)))
    candidates.append(("Struktur Lantai + RSI<45 + AVWAP + RVOL>=0.8", (df["dist_to_low50_pct"] <= 2) & (df["rsi_14"] < 45) & df["close_above_avwap"] & (df["rvol_20"] >= 0.8)))

    results = [evaluate(df, mask, label) for label, mask in candidates]
    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    logger.info("\n%s", results_df.to_string(index=False))


if __name__ == "__main__":
    run()
