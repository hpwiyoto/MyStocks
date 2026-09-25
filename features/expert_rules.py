"""Expert-system-style rule engine for the Momentum Screener page --
direct user request, built from their own trading rules (not backtested
yet, unlike features.momentum_screener's validated_signal combo -- see
this module's HONEST-LABELING note at the bottom before wiring into the
UI).

Rules, as specified by the user, restated precisely:
  1. GOLDEN CROSS, FRESH: SMA50 just crossed above SMA200 (the classic
     "Golden Cross"), OR SMA20 just crossed above SMA50 (a shorter-term
     version) -- "baru" (fresh), not one that happened long ago.
  2. VOLUME CONFIRMATION: above-average volume alongside it.
  3. RSI NORMAL, NOT OVERSOLD: RSI in a healthy middle range, not down in
     oversold territory (a MA cross during an oversold RSI dip reads as
     weak/contradictory, not a confirmed setup).
  4. MACD HISTOGRAM EARLY-STAGE: NOT at a "thick green peak" (already
     extended -- late), NOT fading from green toward red (already
     rolling over) -- instead freshly crossing (or about to cross) the
     zero line from below, still early.
  5. FOREIGN FLOW BONUS (optional, adds confidence, not a gate): net
     foreign accumulation in the trailing few days.
  6. SUPPORT PROXIMITY BONUS (optional, adds confidence, not a gate):
     price sitting close to a recent support level.

Rules 1-4 are GATES (all four must pass for a ticker to appear at all);
5-6 are BONUSES that raise the confidence score but never gate a ticker
out on their own -- exactly the "bisa dijadikan menambah bobot" (CAN be
used to add weight) framing the user used, as opposed to "pastikan"
(make sure) for 1-4.

IMPORTANT CAVEAT, stated up front rather than buried: this project has
directly tested a few of the individual ingredients here before, and
found the OPPOSITE of what a trend-following read like this assumes --
"Golden Cross alone" tested at 26.2% win rate (BELOW the 30.55% null
baseline, scripts/test_strategy_6_criteria.py), and repeatedly across
this project's other momentum research, an already-confirmed-bullish
reading (Bullish MACD status, CMF>0/accumulation) tests WORSE than a
still-weak-looking one, while a still-bearish-looking reading (CMF<0,
bottoming regime) tests better -- see features/momentum_screener.py's
own REGIME_PRIORITY and VALIDATED_RVOL_THRESHOLD comments for the full
history. That doesn't mean THIS specific combination (cross + volume +
RSI + fresh-MACD-not-extended, which is meaningfully more filtered than
a bare golden cross alone) fails the same way -- it hasn't been tested
as combined here -- but it's real, relevant, documented evidence from
this exact codebase that a trend-CONFIRMATION style has tended to
underperform a trend-EXHAUSTION style on this project's own 5-year
historical data. Treat this tier as the user's own trading judgment,
not as proven -- same posture the Momentum Screener page already takes
with its non-validated tiers (divergence, regime priority). Backtest it
the same way (scripts/search_momentum_rules.py's method) before trusting
it the way the validated_signal combo is trusted.

Weights below are a REASONABLE heuristic split of 100 points across how
much each gate/bonus signals confidence (golden cross type matters most,
since MA50/200 is the classical stronger signal over MA20/50; the two
bonuses are worth less since they're optional confirmations) -- NOT
backtest-derived certainty factors (there is no data behind these exact
numbers yet). Documented as a starting point to tune once backtested,
not presented as validated.
"""
import numpy as np
import pandas as pd

GOLDEN_CROSS_LOOKBACK_DAYS = 5   # a crossover counts as "fresh" if it happened within this many trading days
RSI_NORMAL_MIN = 40              # comfortably above oversold (conventional oversold line is 30)
RSI_NORMAL_MAX = 70              # not yet overbought
VOLUME_CONFIRM_RVOL = 1.0        # at/above its own 20-day average volume
MACD_PHASE_LOOKBACK_DAYS = 20    # trailing window used to judge "near a peak" / "near zero"
MACD_PEAK_FRACTION = 0.75        # current hist >= this fraction of its own trailing max -> "already at a peak"
MACD_NEAR_ZERO_FRACTION = 0.35   # while still negative, within this fraction of its trailing range -> "about to cross"
FOREIGN_FLOW_LOOKBACK_DAYS = 5   # trailing days summed for the accumulation bonus
SUPPORT_PROXIMITY_MAX_PCT = 3.0  # within this % of the rolling low (support) counts as "at support"

# Points awarded per rule when it fires -- see module docstring's caveat
# about these being a reasonable starting split, not backtest-derived.
WEIGHT_GOLDEN_CROSS_50_200 = 40
WEIGHT_GOLDEN_CROSS_20_50 = 25
WEIGHT_MACD_EARLY_STAGE = 25
WEIGHT_VOLUME_CONFIRMATION = 15
WEIGHT_RSI_NORMAL = 10
WEIGHT_FOREIGN_FLOW_BONUS = 10
WEIGHT_SUPPORT_PROXIMITY_BONUS = 10
MAX_SCORE = 100


def _detect_cross(fast: np.ndarray, slow: np.ndarray, lookback_days: int) -> tuple[bool, int | None]:
    """fast/slow: ascending-by-date arrays, same length (e.g. sma_20 and
    sma_50). True + days-since if fast crossed above slow within the last
    `lookback_days` trading days AND is STILL above it today -- a same-day
    (or since) whipsaw back below doesn't count as a fresh, still-active
    cross. False, None if fast isn't currently above slow at all, or it is
    but the crossover point itself is further back than the lookback
    window (already-established, not "baru")."""
    n = len(fast)
    if n < 2:
        return False, None
    if np.isnan(fast[-1]) or np.isnan(slow[-1]) or fast[-1] <= slow[-1]:
        return False, None
    start = max(1, n - lookback_days)
    for i in range(n - 1, start - 1, -1):
        if np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]):
            continue
        if fast[i - 1] <= slow[i - 1]:
            return True, n - 1 - i
    return False, None


def classify_macd_phase(macd_hist: np.ndarray, macd_hist_slope_3d: float) -> str:
    """One of "early_bullish" (fresh or about-to cross the zero line --
    the entry the user described), "extended" (already deep positive, a
    peak -- late), "fading" (positive but declining -- rolling over, not
    an entry), "bearish", or "unknown" (insufficient data). A refinement
    of features.momentum_screener.classify_macd_status's plain crossover
    read: that function alone can't distinguish a JUST-crossed histogram
    that's already ballooned large (still technically "Bullish Crossover"
    if the flip happened within its 3-day window, but not what "baru"
    means here) from a genuinely early one.
    """
    hist = macd_hist[~np.isnan(macd_hist)]
    if len(hist) < 2 or pd.isna(macd_hist_slope_3d):
        return "unknown"
    current = hist[-1]
    window = hist[-MACD_PHASE_LOOKBACK_DAYS:]
    recent_max = float(window.max())
    recent_min = float(window.min())
    hist_range = max(recent_max - min(recent_min, 0.0), 1e-9)

    if current > 0:
        if macd_hist_slope_3d < 0:
            return "fading"
        if recent_max > 0 and current >= MACD_PEAK_FRACTION * recent_max:
            return "extended"
        return "early_bullish"
    if macd_hist_slope_3d > 0 and abs(current) <= MACD_NEAR_ZERO_FRACTION * hist_range:
        return "early_bullish"
    return "bearish"


def evaluate_expert_signal(g: pd.DataFrame) -> dict:
    """g: one ticker's rows, ascending by date, with close, volume,
    rsi_14, macd_hist, macd_hist_slope_3d, rvol_20, sma_20, sma_50,
    sma_200, distance_to_support_pct, net_foreign_flow columns (from
    app.data.load_screener_raw_panel). Returns a dict with `passed` (all
    4 gates met), `score` (0-100, only meaningful when passed), `tier`
    ("kuat"/"cukup" when passed), `cross_type`, `macd_phase`, per-rule
    booleans, and `explanation` (list of human-readable strings for the
    UI to show why/why not).
    """
    latest = g.iloc[-1]
    explanation = []

    cross_50_200, days_50_200 = _detect_cross(
        g["sma_50"].to_numpy(dtype=float), g["sma_200"].to_numpy(dtype=float), GOLDEN_CROSS_LOOKBACK_DAYS,
    )
    cross_20_50, days_20_50 = _detect_cross(
        g["sma_20"].to_numpy(dtype=float), g["sma_50"].to_numpy(dtype=float), GOLDEN_CROSS_LOOKBACK_DAYS,
    )
    golden_cross = cross_50_200 or cross_20_50
    if cross_50_200:
        cross_type = "MA50xMA200"
        explanation.append(f"Golden Cross MA50xMA200 -- {days_50_200} hari lalu")
    elif cross_20_50:
        cross_type = "MA20xMA50"
        explanation.append(f"Golden Cross MA20xMA50 -- {days_20_50} hari lalu")
    else:
        cross_type = None
        explanation.append("Belum ada Golden Cross baru (MA50xMA200 atau MA20xMA50)")

    macd_phase = classify_macd_phase(g["macd_hist"].to_numpy(dtype=float), latest.get("macd_hist_slope_3d"))
    macd_ok = macd_phase == "early_bullish"
    macd_reason = {
        "early_bullish": "MACD histogram baru/menjelang crossover -- masih awal",
        "extended": "MACD histogram sudah di puncak -- terlambat, dilewati",
        "fading": "MACD histogram menurun dari hijau -- mulai melemah, dilewati",
        "bearish": "MACD histogram masih negatif dan belum mendekati nol",
        "unknown": "Data MACD histogram belum cukup",
    }[macd_phase]
    explanation.append(macd_reason)

    rvol = latest.get("rvol_20")
    volume_ok = pd.notna(rvol) and rvol >= VOLUME_CONFIRM_RVOL
    explanation.append(
        f"Volume {'terkonfirmasi' if volume_ok else 'BELUM terkonfirmasi'} "
        f"(RVOL {rvol:.2f}x, ambang {VOLUME_CONFIRM_RVOL:.1f}x)" if pd.notna(rvol) else "Data volume relatif tidak ada"
    )

    rsi = latest.get("rsi_14")
    rsi_ok = pd.notna(rsi) and RSI_NORMAL_MIN <= rsi <= RSI_NORMAL_MAX
    if pd.notna(rsi):
        explanation.append(
            f"RSI {rsi:.1f} -- {'normal, tidak oversold' if rsi_ok else f'di luar rentang normal {RSI_NORMAL_MIN}-{RSI_NORMAL_MAX}'}"
        )
    else:
        explanation.append("Data RSI tidak ada")

    passed = golden_cross and macd_ok and volume_ok and rsi_ok

    score = 0
    if passed:
        score += WEIGHT_GOLDEN_CROSS_50_200 if cross_50_200 else WEIGHT_GOLDEN_CROSS_20_50
        score += WEIGHT_MACD_EARLY_STAGE + WEIGHT_VOLUME_CONFIRMATION + WEIGHT_RSI_NORMAL

    flow_window = g["net_foreign_flow"].tail(FOREIGN_FLOW_LOOKBACK_DAYS)
    flow_sum = flow_window.sum(skipna=True) if flow_window.notna().any() else None
    foreign_flow_bonus = passed and flow_sum is not None and flow_sum > 0
    if foreign_flow_bonus:
        score += WEIGHT_FOREIGN_FLOW_BONUS
        explanation.append(f"Bonus: akumulasi net foreign flow {FOREIGN_FLOW_LOOKBACK_DAYS} hari terakhir positif")
    elif flow_sum is None:
        explanation.append("Data foreign flow tidak tersedia untuk ticker ini (tidak mempengaruhi skor)")

    dist_support = latest.get("distance_to_support_pct")
    near_support_bonus = passed and pd.notna(dist_support) and dist_support <= SUPPORT_PROXIMITY_MAX_PCT
    if near_support_bonus:
        score += WEIGHT_SUPPORT_PROXIMITY_BONUS
        explanation.append(f"Bonus: harga dekat support ({dist_support:.1f}% dari level terendah)")

    score = min(score, MAX_SCORE)
    tier = None
    if passed:
        tier = "kuat" if score >= 85 else "cukup"

    return {
        "passed": passed,
        "score": score,
        "tier": tier,
        "cross_type": cross_type,
        "golden_cross": golden_cross,
        "macd_phase": macd_phase,
        "macd_ok": macd_ok,
        "volume_ok": volume_ok,
        "rsi_ok": rsi_ok,
        "foreign_flow_bonus": foreign_flow_bonus,
        "near_support_bonus": near_support_bonus,
        "distance_to_support_pct": dist_support,
        "explanation": explanation,
    }


def compute_expert_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: same long-format frame as features.momentum_screener.
    compute_screener_panel (from app.data.load_screener_raw_panel).
    Returns one row per ticker that PASSED all 4 gates (tickers that
    didn't pass are dropped entirely -- this is a screener, not a
    ranking of everything), sorted by score descending.
    """
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for code, g in panel.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 2:
            continue
        result = evaluate_expert_signal(g)
        if not result["passed"]:
            continue
        latest = g.iloc[-1]
        rows.append({
            "stock_code": code,
            "date": latest["date"],
            "close": latest["close"],
            "rsi_14": latest["rsi_14"],
            "rvol_20": latest["rvol_20"],
            **result,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["score", "stock_code"], ascending=[False, True]).reset_index(drop=True)
