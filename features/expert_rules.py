"""Expert-system-style rule engine for the Momentum Screener page --
direct user request, built from their own trading rules. BACKTESTED
(scripts/backtest_expert_golden_cross.py, 2026-09-25) -- result: NOT
proven to beat doing nothing, unlike features.momentum_screener's
validated_signal combo. See the BACKTEST RESULT note below before
trusting this tier the way validated_signal is trusted.

Rules, as specified by the user, restated precisely:
  1. GOLDEN CROSS, FRESH: SMA50 just crossed above SMA200 (the classic
     "Golden Cross"), OR SMA20 just crossed above SMA50 (a shorter-term
     version) -- "baru" (fresh), not one that happened long ago. OR,
     added on user follow-up: SMA9 is "menjelang" (about to) cross above
     SMA20 -- hasn't crossed yet, but the gap is narrowing and small (an
     earlier, more speculative alternative entry) -- see
     _detect_approaching_cross.
  2. VOLUME CONFIRMATION: above-average volume alongside it.
  3. RSI NORMAL, NOT OVERSOLD: RSI in a healthy middle range, not down in
     oversold territory (a MA cross during an oversold RSI dip reads as
     weak/contradictory, not a confirmed setup).
  4. MACD HISTOGRAM EARLY-STAGE: NOT at a "thick green peak" (already
     extended -- late), NOT fading from green toward red (already
     rolling over) -- instead freshly crossing (or about to cross) the
     zero line from below, still early.
  5. LIQUIDITY: average daily traded value at/above a "cukup likuid"
     threshold -- added on user follow-up ("jangan cuma ikut aturan
     tadi, cari juga yang likuiditasnya baik"), a HARD requirement like
     1-4, not a bonus (illiquid names are a tradability problem, same
     reasoning as excluding suspended/ARA names elsewhere in this app).
  6. FOREIGN FLOW BONUS (optional, adds confidence, not a gate): net
     foreign accumulation in the trailing few days.
  7. SUPPORT PROXIMITY BONUS (optional, adds confidence, not a gate):
     price sitting close to a recent support level.
  8. AVWAP BONUS (optional, adds confidence, not a gate): close >= the
     Anchored VWAP from the rolling 50-day low (same indicator as
     features.momentum_screener.compute_avwap_from_low) -- direct user
     follow-up ("apakah AVWAP bisa dimasukkan ke expert system") after
     AVWAP was REMOVED as a gate from features.momentum_screener.
     is_validated_signal for hurting that (contrarian-style) combo;
     hypothesis was it might fit better with THIS trend-continuation-
     style combo instead. TESTED (see BACKTEST RESULT below): it does
     NOT help here either (n=271, below both the null baseline and the
     combo's own numbers without it) -- kept anyway since it's a
     non-gating, purely additive score/display item with no downside to
     keeping, but treat it as informational, not predictive.
  9. DEEP PULLBACK BONUS (optional, adds confidence, not a gate): direct
     user follow-up, restated precisely after a first pass ("2 bulan
     terakhir") was refined to this exact definition -- "penurunan lebih
     dari 13% dari harga tertinggi setelah rebound terakhir atau setelah
     RSI oversold terakhir". Anchor point = the MORE RECENT of (a) the
     last confirmed swing low in price (a "rebound" --
     features.support_resistance.find_swing_lows, operating on the daily
     low, already used for the Detail Saham chart's S/R lines) or (b)
     the last day RSI was oversold (<RSI_OVERSOLD_THRESHOLD). From that
     anchor onward, take the highest close reached since, and check
     whether the CURRENT close has fallen DEEP_PULLBACK_MIN_PCT (13%) or
     more from that peak. None if neither anchor exists in the available
     window (bonus simply doesn't apply, not an error).

Rules 1-5 are GATES (all five must pass for a ticker to appear at all);
6-9 are BONUSES that raise the confidence score but never gate a ticker
out on their own -- exactly the "bisa dijadikan menambah bobot" (CAN be
used to add weight) framing the user used, as opposed to "pastikan"
(make sure) for 1-5.

BACKTEST RESULT, v3 -- CURRENT 5-gate version, CORRECTED as-of-date
sampling (scripts/backtest_expert_golden_cross.py, 2026-09-25, same
+5%/-2.5%/10-trading-day target as scripts/search_momentum_rules.py, 5
years of IDX history, 120 as-of dates via
scripts.search_momentum_rules.select_as_of_dates -- calendar-anchored,
NOT the old array-position stride v1/v2 below used). Still not
statistically distinguishable from the null baseline -- and the one
subset that looked promising under the old (buggy) sampling now looks
WORSE than null with the fix, confirming it was a sampling artifact,
not a real signal.

    Null baseline:                        n=74,769  WR=32.02%
    FULL COMBO (all 5 gates, what ships): n=387     WR=33.07%  Wilson LB=28.57%  <- below null
    FULL COMBO, 'Kuat' tier (score>=85):  n=340     WR=32.94%  Wilson LB=28.16%
    FULL COMBO + near-support bonus:      n=57      WR=38.60%  Wilson LB=27.06%  <- ALSO below null now (was 35.0% under buggy v2 sampling)
    FULL COMBO + AVWAP bonus:             n=271     WR=31.73%  Wilson LB=26.48%  <- below null AND below the full combo without it
    FULL COMBO + foreign flow bonus:      n=54      WR=37.04%  Wilson LB=25.42%
    FULL COMBO + deep-pullback bonus:     n=7       WR=14.29%  Wilson LB=2.57%   <- n WAY too small, no conclusion possible
    Golden Cross MA50xMA200 alone:        n=1,090   WR=29.63%  Wilson LB=27.00%
    Golden Cross (any type) alone:        n=14,285  WR=31.03%  Wilson LB=30.27%
    MACD early-stage alone:               n=17,138  WR=31.01%  Wilson LB=30.32%
    Volume confirmation alone:            n=24,651  WR=32.05%  Wilson LB=31.47%
    RSI normal alone:                     n=52,709  WR=31.06%  Wilson LB=30.67%

AVWAP BONUS, direct user follow-up ("apakah AVWAP bisa dimasukkan ke
expert system") after AVWAP was found to HURT features.momentum_
screener.is_validated_signal's (contrarian-style) combo -- hypothesis
was that AVWAP might fit better with THIS (trend-continuation-style)
combo instead. Tested (n=271, a reasonably large sample, not a small-n
fluke): it does NOT help here either -- WR=31.73%/LB=26.48%, both below
the null baseline AND below the full combo's own numbers without the
bonus (33.07%/28.57%). Kept in the code anyway (WEIGHT_AVWAP_BONUS,
computed and shown same as the other two bonuses) since it's a
non-gating, purely additive score/display item and removing it buys
nothing safety-wise -- but do not expect it to mean anything predictive.

DEEP PULLBACK BONUS, direct user follow-up ("saham yang sudah
mengalami penurunan diatas 15% selama 2 bulan terakhir", refined to
"lebih dari 13% dari harga tertinggi setelah rebound terakhir atau
setelah RSI oversold terakhir"). Tested: n=7 -- WAY too small to draw
any conclusion, positive or negative (the raw 14.29% win rate LOOKS
bad, but at n=7 that's not statistically distinguishable from noise
either way; Wilson LB=2.57% just reflects how little a 7-observation
sample can rule out). The gate combination this bonus stacks on top of
is already selective (n=387 total), and requiring a confirmed swing-low
or RSI-oversold anchor point on top of that narrows it further --
expect n to grow only slowly over time as more historical instances
accumulate. Kept in the code (non-gating, same reasoning as the other
bonuses) but treat this one as literally untested rather than
"tested and found unhelpful" like AVWAP/near-support/foreign-flow above
-- there just isn't enough data yet to say either way.

Every candidate here sits at or below the null baseline's own Wilson
band -- the FULL COMBO's raw win rate (33.07%) is close to but under
null (32.02%), and its LB (28.57%) is clearly under. The near-support
bonus subset -- which under the OLD, buggy array-position stride (see
scripts.search_momentum_rules.select_as_of_dates's docstring for the
full story) looked like the one genuinely interesting lead (v2: n=57,
WR=47%, LB=35.0%, above null) -- now shows n=57, WR=39%, LB=27.1% with
the corrected sampling, BELOW null. This is a clean, direct
demonstration of exactly why the sampling fix mattered: the same exact
n=57 rows-of-interest under the old method looked like a real edge and
under the fixed method don't -- the underlying market reality didn't
change, only which historical dates got tested.

For context, an EARLIER "Golden Cross" tested in this project
(scripts/test_strategy_6_criteria.py, also below null) was a DIFFERENT
pattern -- MACD-line-crosses-signal-line-below-zero, not this module's
price-moving-average crossover. This backtest is the real test of the
price-MA version, and it also lands at/below null, now confirmed with
corrected methodology too. Consistent with this project's repeated
finding elsewhere (see features/momentum_screener.py's REGIME_PRIORITY/
VALIDATED_RVOL_THRESHOLD comments) that an already-confirmed-bullish
read tends to underperform a still-weak-looking one on this project's
own 5-year IDX data.

Conclusion: ship this as a clearly-unproven manual-observation aid (same
posture as the page's other non-validated tiers -- divergence, regime
priority), NOT as validated. Do not present its score/tier as a
confidence level backed by data; it's a heuristic point count. If
tuned further, re-run scripts/backtest_expert_golden_cross.py rather
than assuming a change helps -- and be wary of testing many small
variations, exactly the multiple-comparisons trap that produced the
misleading n=192 spike scripts/grid_search_momentum_rules.py's docstring
warns about.

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

from features.momentum_screener import compute_avwap_from_low
from features.support_resistance import find_swing_lows

GOLDEN_CROSS_LOOKBACK_DAYS = 5   # a crossover counts as "fresh" if it happened within this many trading days
RSI_NORMAL_MIN = 40              # comfortably above oversold (conventional oversold line is 30)
RSI_NORMAL_MAX = 70              # not yet overbought
VOLUME_CONFIRM_RVOL = 1.0        # at/above its own 20-day average volume
MACD_PHASE_LOOKBACK_DAYS = 20    # trailing window used to judge "near a peak" / "near zero"
MACD_PEAK_FRACTION = 0.75        # current hist >= this fraction of its own trailing max -> "already at a peak"
MACD_NEAR_ZERO_FRACTION = 0.35   # while still negative, within this fraction of its trailing range -> "about to cross"
FOREIGN_FLOW_LOOKBACK_DAYS = 5   # trailing days summed for the accumulation bonus
SUPPORT_PROXIMITY_MAX_PCT = 3.0  # within this % of the rolling low (support) counts as "at support"

# MA9xMA20 "menjelang" (about to cross, NOT crossed yet) -- direct user
# follow-up request, an earlier/more speculative alternative entry to the
# two already-crossed Golden Cross types above. SMA9 isn't a column
# feature_daily stores (only ema_9), so it's computed on the fly from the
# same window's close prices -- see evaluate_expert_signal -- rather than
# widening the daily feature pipeline for one extra rolling average.
MA9_APPROACH_LOOKBACK_DAYS = 5   # the gap must have been narrowing over this many trading days
MA9_APPROACH_GAP_MAX_PCT = 2.0   # SMA9 within this % of SMA20 (from below) counts as "menjelang"

# Liquidity gate -- direct user follow-up request ("jangan cuma ikut
# aturan tadi, cari juga yang likuiditasnya baik"): a HARD requirement
# (like gates 1-4), not a soft bonus, since illiquid names are a
# tradability problem the same way a suspended/ARA-excluded ticker is
# (see app.data.load_suspended_tickers), not just a nice-to-have.
# Computed from THIS window's own close*volume (not a separate call to
# app.data.load_liquidity, which averages over a different window and
# isn't available inside a historical backtest replay) so this function
# stays self-contained and gives IDENTICAL behavior live and backtested.
# Threshold matches this project's own existing "cukup likuid" tier (see
# app.style.LIQUIDITY_FILTER_OPTIONS' "≥ Rp 1 miliar/hari" option) rather
# than inventing a new number.
LIQUIDITY_WINDOW_DAYS = 20
MIN_AVG_TRADED_VALUE = 1_000_000_000  # Rp 1 miliar/hari

# Deep pullback bonus -- direct user follow-up, exact definition: decline
# from the highest close since the more recent of (a) the last confirmed
# swing low ("rebound") or (b) the last RSI-oversold day. See rule 9's
# docstring note above and _deep_pullback_pct below.
DEEP_PULLBACK_MIN_PCT = 13.0
RSI_OVERSOLD_THRESHOLD = 30.0  # conventional oversold line, matches RSI_NORMAL_MIN's own comment

# Points awarded per rule when it fires -- see module docstring's caveat
# about these being a reasonable starting split, not backtest-derived.
WEIGHT_GOLDEN_CROSS_50_200 = 40
WEIGHT_GOLDEN_CROSS_20_50 = 25
WEIGHT_MA9_APPROACHING = 15      # lower than either confirmed cross -- this one hasn't happened yet
WEIGHT_MACD_EARLY_STAGE = 25
WEIGHT_VOLUME_CONFIRMATION = 15
WEIGHT_RSI_NORMAL = 10
WEIGHT_LIQUIDITY = 10
WEIGHT_FOREIGN_FLOW_BONUS = 10
WEIGHT_SUPPORT_PROXIMITY_BONUS = 10
WEIGHT_AVWAP_BONUS = 10
WEIGHT_DEEP_PULLBACK_BONUS = 10
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


def _detect_approaching_cross(fast: np.ndarray, slow: np.ndarray, lookback_days: int, gap_max_pct: float) -> tuple[bool, float | None]:
    """The NOT-yet-crossed counterpart to _detect_cross: True + the
    current gap (%) if `fast` is still below `slow` today, but the gap
    has been narrowing over the last `lookback_days` trading days AND is
    now within `gap_max_pct` -- an earlier, more speculative "about to
    cross" read (menjelang) rather than _detect_cross's already-happened
    one. False, None if fast is already >= slow (that's a completed
    cross, not "menjelang" anymore -- _detect_cross's job), or there
    isn't enough history, or the gap isn't actually narrowing (could be
    close by chance while still diverging)."""
    n = len(fast)
    if n < lookback_days + 1:
        return False, None
    if np.isnan(fast[-1]) or np.isnan(slow[-1]) or slow[-1] <= 0 or fast[-1] >= slow[-1]:
        return False, None
    gap_now = (slow[-1] - fast[-1]) / slow[-1] * 100
    prev = n - 1 - lookback_days
    if np.isnan(fast[prev]) or np.isnan(slow[prev]) or slow[prev] <= 0:
        return False, None
    gap_before = (slow[prev] - fast[prev]) / slow[prev] * 100
    approaching = gap_now < gap_before and 0 <= gap_now <= gap_max_pct
    return approaching, gap_now


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


def _last_rebound_anchor_idx(low: np.ndarray, rsi: np.ndarray) -> int | None:
    """Index of the MORE RECENT of: the last confirmed swing low in price
    (a "rebound" -- features.support_resistance.find_swing_lows, which
    requires a full trailing window AFTER a point before it counts as
    confirmed, so a dip that hasn't turned around yet won't show up) or
    the last day RSI was oversold (<RSI_OVERSOLD_THRESHOLD). This is the
    anchor point for the deep-pullback bonus, per the user's own
    definition ("dari harga tertinggi setelah rebound terakhir atau
    setelah RSI oversold terakhir"). None if neither exists in the
    available window."""
    swing_lows = find_swing_lows(low)
    last_swing_low = swing_lows[-1] if swing_lows else None
    oversold_idx = np.where(rsi < RSI_OVERSOLD_THRESHOLD)[0]
    last_oversold = int(oversold_idx[-1]) if len(oversold_idx) else None
    candidates = [i for i in (last_swing_low, last_oversold) if i is not None]
    return max(candidates) if candidates else None


def _deep_pullback_pct(close: np.ndarray, low: np.ndarray, rsi: np.ndarray) -> float | None:
    """% decline from the highest close since _last_rebound_anchor_idx to
    the current (last) close. None if no anchor point exists yet."""
    anchor = _last_rebound_anchor_idx(low, rsi)
    if anchor is None:
        return None
    peak = float(close[anchor:].max())
    if peak <= 0 or np.isnan(peak) or np.isnan(close[-1]):
        return None
    return (peak - float(close[-1])) / peak * 100


def evaluate_expert_signal(g: pd.DataFrame) -> dict:
    """g: one ticker's rows, ascending by date, with close, high, low,
    volume, rsi_14, macd_hist, macd_hist_slope_3d, rvol_20, sma_20,
    sma_50, sma_200, distance_to_support_pct, net_foreign_flow columns
    (from app.data.load_screener_raw_panel -- high/low needed for the
    AVWAP bonus's compute_avwap_from_low call and the deep-pullback
    bonus's swing-low detection). Returns a dict with `passed` (all 5
    gates met), `score` (0-100, only meaningful when passed), `tier`
    ("kuat"/"cukup" when passed), `cross_type`, `macd_phase`, per-rule
    booleans (including the 4 bonuses: foreign_flow_bonus,
    near_support_bonus, avwap_bonus, deep_pullback_bonus), and
    `explanation` (list of human-readable strings for the UI to show
    why/why not).
    """
    latest = g.iloc[-1]
    explanation = []

    cross_50_200, days_50_200 = _detect_cross(
        g["sma_50"].to_numpy(dtype=float), g["sma_200"].to_numpy(dtype=float), GOLDEN_CROSS_LOOKBACK_DAYS,
    )
    cross_20_50, days_20_50 = _detect_cross(
        g["sma_20"].to_numpy(dtype=float), g["sma_50"].to_numpy(dtype=float), GOLDEN_CROSS_LOOKBACK_DAYS,
    )
    # SMA9 isn't a stored feature_daily column -- computed here from this
    # window's own close prices (see MA9_APPROACH_* constants' comment).
    sma_9 = g["close"].astype(float).rolling(9).mean().to_numpy()
    approaching_9_20, gap_9_20 = _detect_approaching_cross(
        sma_9, g["sma_20"].to_numpy(dtype=float), MA9_APPROACH_LOOKBACK_DAYS, MA9_APPROACH_GAP_MAX_PCT,
    )

    golden_cross = cross_50_200 or cross_20_50 or approaching_9_20
    if cross_50_200:
        cross_type = "MA50xMA200"
        explanation.append(f"Golden Cross MA50xMA200 -- {days_50_200} hari lalu")
    elif cross_20_50:
        cross_type = "MA20xMA50"
        explanation.append(f"Golden Cross MA20xMA50 -- {days_20_50} hari lalu")
    elif approaching_9_20:
        cross_type = "MA9xMA20_menjelang"
        explanation.append(f"MA9 menjelang cross MA20 -- selisih {gap_9_20:.1f}%, sedang menyempit")
    else:
        cross_type = None
        explanation.append("Belum ada Golden Cross baru atau MA9 menjelang cross MA20")

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

    liq_window = g.tail(LIQUIDITY_WINDOW_DAYS)
    traded_value = liq_window["close"].astype(float) * liq_window["volume"].astype(float)
    avg_traded_value = float(traded_value.mean()) if not traded_value.empty else None
    liquidity_ok = avg_traded_value is not None and avg_traded_value >= MIN_AVG_TRADED_VALUE
    if avg_traded_value is not None:
        explanation.append(
            f"Likuiditas {'cukup' if liquidity_ok else 'BELUM cukup'} "
            f"(rata-rata Rp {avg_traded_value / 1e9:.1f} M/hari, ambang Rp {MIN_AVG_TRADED_VALUE / 1e9:.0f} M/hari)"
        )
    else:
        explanation.append("Data volume transaksi tidak ada")

    passed = golden_cross and macd_ok and volume_ok and rsi_ok and liquidity_ok

    score = 0
    if passed:
        if cross_50_200:
            score += WEIGHT_GOLDEN_CROSS_50_200
        elif cross_20_50:
            score += WEIGHT_GOLDEN_CROSS_20_50
        else:
            score += WEIGHT_MA9_APPROACHING
        score += WEIGHT_MACD_EARLY_STAGE + WEIGHT_VOLUME_CONFIRMATION + WEIGHT_RSI_NORMAL + WEIGHT_LIQUIDITY

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

    avwap = compute_avwap_from_low(
        g["close"].to_numpy(dtype=float), g["high"].to_numpy(dtype=float),
        g["low"].to_numpy(dtype=float), g["volume"].to_numpy(dtype=float),
    )
    close_above_avwap = bool(latest["close"] >= avwap) if avwap is not None and pd.notna(latest["close"]) else None
    avwap_bonus = passed and close_above_avwap is True
    if avwap_bonus:
        score += WEIGHT_AVWAP_BONUS
        explanation.append("Bonus: harga di atas Anchored VWAP dari titik terendah 50 hari")
    elif close_above_avwap is None:
        explanation.append("Data AVWAP tidak cukup (tidak mempengaruhi skor)")

    pullback_pct = _deep_pullback_pct(
        g["close"].to_numpy(dtype=float), g["low"].to_numpy(dtype=float), g["rsi_14"].to_numpy(dtype=float),
    )
    deep_pullback_bonus = passed and pullback_pct is not None and pullback_pct >= DEEP_PULLBACK_MIN_PCT
    if deep_pullback_bonus:
        score += WEIGHT_DEEP_PULLBACK_BONUS
        explanation.append(f"Bonus: turun {pullback_pct:.1f}% dari puncak sejak rebound/RSI oversold terakhir")
    elif pullback_pct is None:
        explanation.append("Belum ada rebound/RSI oversold terkonfirmasi dalam jendela data (tidak mempengaruhi skor)")

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
        "ma9_approaching_gap_pct": gap_9_20,
        "macd_phase": macd_phase,
        "macd_ok": macd_ok,
        "volume_ok": volume_ok,
        "rsi_ok": rsi_ok,
        "liquidity_ok": liquidity_ok,
        "avg_traded_value": avg_traded_value,
        "foreign_flow_bonus": foreign_flow_bonus,
        "near_support_bonus": near_support_bonus,
        "distance_to_support_pct": dist_support,
        "avwap_bonus": avwap_bonus,
        "close_above_avwap": close_above_avwap,
        "deep_pullback_bonus": deep_pullback_bonus,
        "deep_pullback_pct": pullback_pct,
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
