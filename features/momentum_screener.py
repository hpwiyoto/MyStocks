"""Manual/discretionary technical screener -- deliberately NOT model-driven
(unlike engine/predict.py, engine/predict_turnaround.py). Classifies MACD's
crossover + zero-line + histogram-momentum status (the 3-part reading the
user was taught) and detects RSI/MACD bullish divergence against the two
most recent price swing lows, feeding app/pages/4_📡_Momentum_Screener.py's
filters. MACD itself was empirically tested and rejected as a MODEL feature
(see scripts/test_macd_zscore_feature.py) -- this module is the other,
approved use: a rule-based filter a trader reads directly, no ML involved,
model probability shown only as a secondary reference column afterward.
"""
import numpy as np
import pandas as pd

FRESH_CROSSOVER_DAYS = 3  # a crossover counts as "fresh" if it happened within this many trading days
SWING_WINDOW = 4          # a close must be the lowest within +/-4 trading days to count as a swing low
MIN_GAP_BETWEEN_LOWS = 5  # trading days -- avoid picking two lows out of the same dip
RECENT_LOW_MAX_AGE = 20   # trading days -- a divergence only counts if its most recent leg is this fresh

# Regime ranking for the sort order below -- REVISED after
# scripts/search_momentum_rules.py backtested every regime against 5 years
# of real outcomes (same +5%/-2.5%/10-trading-day grading as Swing).
# The ORIGINAL version of this ranking put early_reversal/bullish first on
# theoretical grounds ("a fresher trigger, more upside runway than an
# already-confirmed uptrend") -- the backtest showed that reasoning was
# backwards for this specific target: early_reversal (26.6% win rate) and
# bullish (28.2%) both underperform the unconditional null baseline
# (30.55%, n=76,442), while bottoming (35.7%) and bearish (31.5%) --
# BAD_REGIMES, ranked lowest before -- both beat it. Likely explanation: a
# stock already in early_reversal/bullish has already captured its recent
# upside, leaving less room to still rise 5% more in 10 days before a
# -2.5% pullback; a still-"bad" regime that manages to move has more of
# that room left -- a mean-reversion dynamic, not the momentum-continuation
# story the original ordering assumed. Order below now follows the
# observed win rates directly (bottoming > bearish > sideways >
# accumulation > bullish > early_reversal), overextended still last
# (a pullback warning by definition, not tested standalone but has no
# plausible case for ranking above anything here).
REGIME_PRIORITY = {
    "bottoming": 0,
    "bearish": 1,
    "sideways": 2,
    "accumulation": 3,
    "bullish": 4,
    "early_reversal": 5,
    "overextended": 6,
}
DEFAULT_REGIME_PRIORITY = len(REGIME_PRIORITY)  # unknown/missing regime sorts last

# The combination scripts/search_momentum_rules.py + a follow-up 5,880-
# combination grid search (scripts/grid_search_momentum_rules.py) found
# with a real, statistically-supported edge over the null baseline (30.55%
# across 76,442 resolved historical instances) -- NOT the same target
# divergence/regime-priority above were designed around (which turned out,
# per the same backtest, to be indistinguishable from noise:
# scripts/backtest_momentum_screener.py).
#
# The grid's #1 result by Wilson lower bound (accumulation+RVOL>=1.2+
# CMF<0+RSI 30-50, LB=37.4%) was deliberately NOT adopted -- with 5,880
# combinations tested, a handful of small-n spikes (that one: n=192) at
# the very top are exactly what multiple-comparisons overfitting looks
# like, not necessarily a genuinely stronger pattern. This rule instead:
# bottoming + momentum menguat + CMF<0 + RVOL>=0.8 (n=958, win_rate=39.8%,
# Wilson LB=36.7%) -- chosen because it beats the original hand-picked
# rule (bottoming+momentum+RVOL>=1.2, n=640, LB=36.4%) on BOTH axes that
# matter: a HIGHER lower bound AND ~50% more real-world coverage (from
# RELAXING the volume bar to 0.8, not raising it), which is the opposite
# of what an overfit spike looks like. CMF<0 (distribution, not
# accumulation) is counter-intuitive but consistent with every other
# finding in this project's momentum work: still-bearish-looking readings
# (this, plus the bottoming regime itself, plus MACD status not mattering)
# keep correlating with MORE forward room than already-confirmed-positive
# ones -- a stock the market is still selling has more room to surprise
# than one it has already bid up.
VALIDATED_RVOL_THRESHOLD = 0.8

# Added after scripts/test_strategy_6_criteria.py backtested a user-
# proposed 6-criteria strategy (Anchored VWAP among them) against the same
# 76,442-instance historical dataset. The 6-criteria combo AS SPECIFIED
# was unusable stacked together (n=12, win_rate 25% -- WORSE than the
# 30.55% null baseline), and 5 of its 6 pieces were weak-to-harmful in
# isolation (Anchored VWAP alone: 29.1%, Golden Cross alone: 26.2%, both
# below null). But requiring close >= this Anchored VWAP ON TOP OF the
# already-validated rule below genuinely helped: n=958->523,
# win_rate 39.8%->42.4%, Wilson LB 36.7%->38.3% -- a real gain on the
# metric that matters (LB, which penalizes the smaller n), not just a
# smaller sample cherry-picked for a better point estimate.
AVWAP_WINDOW = 50


def compute_avwap_from_low(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> float | None:
    """Anchored VWAP for the LAST bar in the given arrays (ascending by
    date), anchored at the date the rolling AVWAP_WINDOW-day low occurred
    -- an approximation of "the average price smart money has actually
    paid" since that low, using typical price (H+L+C)/3 as the standard
    VWAP price input. Returns None if there's not enough data or zero
    volume throughout the anchored window (can't divide by it)."""
    n = len(close)
    if n == 0:
        return None
    start = max(0, n - AVWAP_WINDOW)
    anchor = start + int(np.argmin(close[start:]))
    seg_vol = volume[anchor:]
    vol_sum = seg_vol.sum()
    if vol_sum <= 0:
        return None
    seg_typical = (high[anchor:] + low[anchor:] + close[anchor:]) / 3
    return float((seg_typical * seg_vol).sum() / vol_sum)


def is_validated_signal(regime, macd_hist_slope_3d, cmf_20, rvol_20, close_above_avwap) -> bool:
    return (
        regime == "bottoming"
        and pd.notna(macd_hist_slope_3d) and macd_hist_slope_3d > 0
        and pd.notna(cmf_20) and cmf_20 < 0
        and pd.notna(rvol_20) and rvol_20 >= VALIDATED_RVOL_THRESHOLD
        and close_above_avwap is True
    )


def classify_macd_status(macd_hist: pd.Series, fresh_days: int = FRESH_CROSSOVER_DAYS) -> str:
    """Crossover status (method 1 of the MACD reading): Bullish Crossover
    (histogram just flipped positive), Bullish (already positive, no fresh
    flip), Bearish Crossover, Bearish, or Netral (exactly zero/insufficient
    data). Zero-line position and histogram momentum are reported
    separately (raw macd sign, and macd_hist_slope_3d respectively) -- kept
    as distinct columns rather than folded into this one category, matching
    how the three methods were taught as separate readings."""
    hist = macd_hist.dropna()
    if hist.empty:
        return "Tidak diketahui"
    current = hist.iloc[-1]
    window = hist.iloc[-(fresh_days + 1):].to_numpy()
    sign = np.sign(window)
    crossed_up = sign[-1] > 0 and (sign[:-1] <= 0).any()
    crossed_down = sign[-1] < 0 and (sign[:-1] >= 0).any()
    if current > 0:
        return "Bullish Crossover" if crossed_up else "Bullish"
    if current < 0:
        return "Bearish Crossover" if crossed_down else "Bearish"
    return "Netral"


def _find_swing_lows(close: np.ndarray, window: int = SWING_WINDOW, min_gap: int = MIN_GAP_BETWEEN_LOWS) -> list[int]:
    """Indices where `close` is the minimum within its own +/-window
    neighborhood, deduped so two candidates from the same dip (closer than
    min_gap trading days apart) collapse into whichever is the deeper low."""
    n = len(close)
    candidates = [
        i for i in range(window, n - window)
        if close[i] == close[i - window: i + window + 1].min()
    ]
    lows: list[int] = []
    for i in candidates:
        if not lows or i - lows[-1] >= min_gap:
            lows.append(i)
        elif close[i] < close[lows[-1]]:
            lows[-1] = i
    return lows


def detect_bullish_divergence(g: pd.DataFrame) -> dict:
    """g: one ticker's rows, ascending by date, with close/rsi_14/macd_hist
    columns. Bullish divergence = price makes a LOWER low while the
    indicator (RSI and/or MACD histogram) makes a HIGHER low across the two
    most recent price swing lows -- the classic "selling pressure exhausted"
    signal. Reports each indicator independently so a caller can tell single
    from double (RSI + MACD together, the strongest form) divergence."""
    close = g["close"].to_numpy(dtype=float)
    lows = _find_swing_lows(close)
    result = {"divergence_rsi": False, "divergence_macd": False, "divergence_age_days": None}
    if len(lows) < 2:
        return result
    low2, low1 = lows[-1], lows[-2]  # low2 = more recent
    age = len(close) - 1 - low2
    if age > RECENT_LOW_MAX_AGE or close[low2] >= close[low1]:
        return result  # too stale, or price didn't actually make a lower low
    rsi = g["rsi_14"].to_numpy(dtype=float)
    macd_hist = g["macd_hist"].to_numpy(dtype=float)
    if not (np.isnan(rsi[low1]) or np.isnan(rsi[low2])) and rsi[low2] > rsi[low1]:
        result["divergence_rsi"] = True
    if not (np.isnan(macd_hist[low1]) or np.isnan(macd_hist[low2])) and macd_hist[low2] > macd_hist[low1]:
        result["divergence_macd"] = True
    if result["divergence_rsi"] or result["divergence_macd"]:
        result["divergence_age_days"] = int(age)
    return result


def compute_screener_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: long-format rows (stock_code, date, close, high, low, volume,
    rsi_14, macd, macd_signal, macd_hist, macd_hist_slope_3d, cmf_20,
    rvol_20), ideally 60+ trading days per ticker (from
    app.data.load_screener_raw_panel; high/low needed for the Anchored
    VWAP check below). Returns one summary row per ticker: latest reading
    + macd_status + divergence flags + a priority tier (0=double
    divergence, 1=single, 2=none) for the screener page to sort by ahead
    of model probability."""
    if panel.empty:
        return pd.DataFrame()

    rows = []
    for code, g in panel.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        latest = g.iloc[-1]
        div = detect_bullish_divergence(g)
        avwap = compute_avwap_from_low(
            g["close"].to_numpy(dtype=float), g["high"].to_numpy(dtype=float),
            g["low"].to_numpy(dtype=float), g["volume"].to_numpy(dtype=float),
        )
        close_above_avwap = bool(latest["close"] >= avwap) if avwap is not None and pd.notna(latest["close"]) else None
        rows.append({
            "stock_code": code,
            "date": latest["date"],
            "close": latest["close"],
            "volume": latest["volume"],
            "rsi_14": latest["rsi_14"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "macd_hist": latest["macd_hist"],
            "macd_hist_slope_3d": latest["macd_hist_slope_3d"],
            "macd_status": classify_macd_status(g["macd_hist"]),
            "macd_above_zero": bool(latest["macd"] > 0) if pd.notna(latest["macd"]) else None,
            "cmf_20": latest["cmf_20"],
            "rvol_20": latest["rvol_20"],
            "regime": latest.get("regime"),
            "avwap_from_low50": avwap,
            "close_above_avwap": close_above_avwap,
            **div,
        })
    out = pd.DataFrame(rows)
    out["divergence_tier"] = np.select(
        [out["divergence_rsi"] & out["divergence_macd"], out["divergence_rsi"] | out["divergence_macd"]],
        [0, 1], default=2,
    )
    out["regime_priority"] = out["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
    out["validated_signal"] = out.apply(
        lambda r: is_validated_signal(
            r["regime"], r["macd_hist_slope_3d"], r["cmf_20"], r["rvol_20"], r["close_above_avwap"],
        ), axis=1,
    )
    # RSI beats MACD as a ranking signal here, backed by two independent
    # findings elsewhere in this project: rsi_distance_50 is a top-3
    # contributor in BOTH the Swing and Turnaround models' own feature-gain
    # ranking, while MACD -- even correctly z-score normalized -- moved
    # ROC-AUC by less than fold-to-fold noise when tested directly (see
    # scripts/test_macd_zscore_feature.py). Consistent with RSI being a
    # leading oscillator and MACD a lagging one. Distance from 50 (not
    # signed, and NOT the model's own rsi_distance_50 -- this is a display/
    # ranking-only value) rewards a stock sitting right at the pivot
    # (crossing out of weakness, the same 45-65-with-rising-slope zone
    # features.regime's early_reversal itself looks for) over one still
    # deep in oversold territory with no confirmed turn yet, or one already
    # closer to this screener's overbought filter edge.
    out["rsi_pivot_distance"] = (out["rsi_14"] - 50).abs()
    return out
