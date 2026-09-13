"""Wyckoff market-cycle status: a SIMPLIFIED, single-bar-computable proxy
for the classic Wyckoff schematic (Accumulation -> Markup -> Distribution
-> Markdown, with Spring/Upthrust as the signature "false break, quick
reclaim" tests within Accumulation/Distribution) -- prompted directly by
a user question ("apakah bisa ditambahkan status dari Wyckoff theory").

Full Wyckoff analysis (Phase A-E, Preliminary Support/Supply, Automatic
Rally/Reaction, Sign of Strength/Weakness, Last Point of Support/Supply,
etc.) is a multi-bar, partly subjective SEQUENCE of events that even
experienced practitioners read differently bar-by-bar -- not something
this project claims to fully formalize. What's implemented here is the
same kind of deliberate simplification as features/support_resistance.py's
rolling-window Fibonacci proxy or the AVWAP "recent swing" window: a
single trailing lookback window stands in for a properly-identified
trading range, so every row is computable with NO lookahead (only past +
current-bar data), same convention as every other feature in this
project.

Definitions -- deliberately EXHAUSTIVE over exactly 4 phases (direct user
request: "gunakan hanya 4 fase saja"), no 5th "indeterminate" bucket the
way the first version had (that version required a near-range-boundary
condition for markup/markdown on top of ranging/trend, which left a large
share of rows -- price that's actively trending but not yet near either
edge of its own window -- unclassified):
- A trailing WYCKOFF_WINDOW-day high/low stands in for "the trading
  range" (a fixed window, not a detected swing/pivot range, for the same
  no-lookahead simplicity as features/test_fibonacci_feature.py's window).
- "Ranging" = that window's width, as a %% of price, sits in the bottom
  WYCKOFF_RANGE_RANK_THRESHOLD percentile of its own trailing
  WYCKOFF_RANK_WINDOW-day history -- compressed volatility, the
  precondition for calling a stretch of price action a Wyckoff "cause-
  building" range at all (same rank-based compression idea as
  features/regime.py's own accumulation rule, generalized to also cover
  distribution, which regime.py's regime classifier does NOT have a
  matching label for). Every row is either ranging or not -- exhaustive
  by construction, since rank() always returns a value once warmed up.
- Ranging -> accumulation (prior trend down) or distribution (prior trend
  up), by the SIGN of the trend in the PRIOR_TREND_WINDOW days ending
  where the current range window starts -- no minimum-magnitude gate, so
  every ranging row lands in one or the other (a dead-flat prior trend,
  sign exactly zero, defaults to distribution, an arbitrary but rare
  tie-break -- ranging is already a real-valued rank comparison, so an
  exact zero prior trend is a measure-zero edge case in practice).
- NOT ranging (i.e. actively trending) -> markup (price now above where
  it was WYCKOFF_WINDOW days ago) or markdown (below) -- this is the
  piece that changed from the first version: no longer gated on ALSO
  being near the range's own high/low, which is what created the old
  "indeterminate" majority.
- spring / upthrust: today's low/high pokes past the PRIOR day's already-
  established range boundary (shift(1), so today's own extreme can't
  inflate the boundary it's being compared to) by SPRING_TOLERANCE_PCT,
  but today's CLOSE recovers back inside the range -- the single-bar
  proxy for Wyckoff's signature "shakeout that fails to follow through"
  test. Not gated on a volume climax (a real Wyckoff Spring/Upthrust is
  usually climactic) -- kept as a separate, simpler flag here; volume
  confirmation is left to scripts/test_wyckoff_feature.py's own A/B test
  to check whether it actually adds anything empirically, same "test
  before adding complexity" posture as everywhere else in this project.

Usage:
    from features.wyckoff import compute_wyckoff_features
    wyckoff_df = compute_wyckoff_features(prices)  # prices: stock_code, date, open, high, low, close, volume
"""
import numpy as np
import pandas as pd

WYCKOFF_WINDOW = 50          # trading days -- matches this project's existing "recent swing" convention
PRIOR_TREND_WINDOW = 50      # trading days immediately before the range window
WYCKOFF_RANK_WINDOW = 100    # trailing history the range-width percentile is computed against
RANGING_RANK_THRESHOLD = 0.35  # bottom 35% width percentile = "compressed enough to call a range"
SPRING_TOLERANCE_PCT = 1.5   # matches features/support_resistance.py's DEFAULT_TOLERANCE_PCT for "a level"

NEW_COLS = ["wyckoff_phase", "wyckoff_spring", "wyckoff_upthrust", "wyckoff_range_position_pct"]


def compute_wyckoff_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["stock_code", "date"]).copy()
    g = prices.groupby("stock_code")

    range_high = g["high"].transform(lambda s: s.rolling(WYCKOFF_WINDOW, min_periods=WYCKOFF_WINDOW).max())
    range_low = g["low"].transform(lambda s: s.rolling(WYCKOFF_WINDOW, min_periods=WYCKOFF_WINDOW).min())
    range_width_pct = (range_high - range_low) / prices["close"] * 100

    # rank(pct=True) needs a real (non-grouped-transform-safe) rolling call
    # per ticker -- same pattern as features/regime.py's bb_rank.
    range_width_rank = (
        range_width_pct.groupby(prices["stock_code"])
        .transform(lambda s: s.rolling(WYCKOFF_RANK_WINDOW, min_periods=WYCKOFF_RANK_WINDOW).rank(pct=True))
    )
    is_ranging = range_width_rank < RANGING_RANK_THRESHOLD

    # Trend in the PRIOR_TREND_WINDOW days ending right where the current
    # range window starts -- shift(WYCKOFF_WINDOW) anchors "today" back to
    # the range's own start, shift(WYCKOFF_WINDOW + PRIOR_TREND_WINDOW)
    # anchors PRIOR_TREND_WINDOW days before that, so this never overlaps
    # the range being classified. Used to split RANGING rows into
    # accumulation vs distribution.
    close_at_range_start = g["close"].transform(lambda s: s.shift(WYCKOFF_WINDOW))
    close_before_range_start = g["close"].transform(lambda s: s.shift(WYCKOFF_WINDOW + PRIOR_TREND_WINDOW))
    prior_trend_pct = (close_at_range_start - close_before_range_start) / close_before_range_start * 100

    # Trend over the CURRENT WYCKOFF_WINDOW itself (today vs WYCKOFF_WINDOW
    # days ago) -- used to split NOT-ranging (actively trending) rows into
    # markup vs markdown. A different window than prior_trend_pct above on
    # purpose: this one describes the move happening RIGHT NOW, not what
    # preceded it.
    recent_trend_pct = (prices["close"] - close_at_range_start) / close_at_range_start * 100

    range_position_pct = np.where(
        range_high > range_low, (prices["close"] - range_low) / (range_high - range_low) * 100, np.nan,
    )

    accumulation = is_ranging & (prior_trend_pct <= 0)
    distribution = is_ranging & (prior_trend_pct > 0)
    markup = ~is_ranging & (recent_trend_pct > 0)
    markdown = ~is_ranging & (recent_trend_pct <= 0)

    conditions = [accumulation, distribution, markup, markdown]
    choices = ["accumulation", "distribution", "markup", "markdown"]
    # No default/"indeterminate" bucket -- the 4 conditions above already
    # exhaustively cover every row where the inputs are non-NaN (is_ranging
    # is always True or False once warmed up; whichever trend measure
    # applies is always > 0 or <= 0). default="" here only ever fires on
    # the warmup rows the has_data mask below blanks out anyway.
    phase = pd.Series(np.select(conditions, choices, default=""), index=prices.index)

    required = [range_high, range_low, range_width_rank, prior_trend_pct, recent_trend_pct]
    has_data = pd.concat(required, axis=1).notna().all(axis=1)
    phase = phase.where(has_data)

    # Spring/upthrust: compare TODAY's low/high against YESTERDAY's already-
    # established range boundary (grouped shift(1), so a ticker's first row
    # never picks up the previous ticker's last value) -- today's own bar
    # can't inflate the very boundary it's being tested against.
    prior_range_low = range_low.groupby(prices["stock_code"]).shift(1)
    prior_range_high = range_high.groupby(prices["stock_code"]).shift(1)
    spring = (prices["low"] < prior_range_low * (1 - SPRING_TOLERANCE_PCT / 100)) & (prices["close"] > prior_range_low)
    upthrust = (prices["high"] > prior_range_high * (1 + SPRING_TOLERANCE_PCT / 100)) & (prices["close"] < prior_range_high)
    spring = spring.astype(float).where(prior_range_low.notna())
    upthrust = upthrust.astype(float).where(prior_range_high.notna())

    out = prices[["stock_code", "date"]].copy()
    out["wyckoff_phase"] = phase
    out["wyckoff_spring"] = spring
    out["wyckoff_upthrust"] = upthrust
    out["wyckoff_range_position_pct"] = range_position_pct
    return out
