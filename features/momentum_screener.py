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

# How much confirmed bullish momentum each features.regime.classify_regime()
# state represents, lowest = most advanced/actionable -- used as a secondary
# sort key on the Momentum Screener page (after divergence_tier, before
# divergence freshness). early_reversal ranks ABOVE bullish deliberately:
# this screener's whole spirit (like divergence-age priority above it) is
# catching a stock AS it turns, not after it's already had its run --
# early_reversal has a real, just-happened trigger (EMA20 reclaim + RSI
# crossing above 50, see features/regime.py) with more upside runway left,
# while bullish is already a mature, sustained uptrend. accumulation sits
# below both: it's a quiet, low-volatility phase with NO confirmed
# directional trigger yet (could resolve either way), so it's a weaker
# signal than either regime that already shows real upward momentum --
# directly answers the "accumulation vs early_reversal" question asked
# when this was built. overextended ranks last among non-bad regimes since
# it's a pullback warning, not a buy setup.
REGIME_PRIORITY = {
    "early_reversal": 0,
    "bullish": 1,
    "accumulation": 2,
    "sideways": 3,
    "bottoming": 4,
    "bearish": 5,
    "overextended": 6,
}
DEFAULT_REGIME_PRIORITY = len(REGIME_PRIORITY)  # unknown/missing regime sorts last


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
    """panel: long-format rows (stock_code, date, close, volume, rsi_14,
    macd, macd_signal, macd_hist, macd_hist_slope_3d, cmf_20, rvol_20),
    ideally 60+ trading days per ticker (from app.data.load_screener_raw_panel).
    Returns one summary row per ticker: latest reading + macd_status +
    divergence flags + a priority tier (0=double divergence, 1=single,
    2=none) for the screener page to sort by ahead of model probability."""
    if panel.empty:
        return pd.DataFrame()

    rows = []
    for code, g in panel.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        latest = g.iloc[-1]
        div = detect_bullish_divergence(g)
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
            **div,
        })
    out = pd.DataFrame(rows)
    out["divergence_tier"] = np.select(
        [out["divergence_rsi"] & out["divergence_macd"], out["divergence_rsi"] | out["divergence_macd"]],
        [0, 1], default=2,
    )
    out["regime_priority"] = out["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
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
