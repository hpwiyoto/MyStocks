"""Market structure features: swing-based higher-high/lower-low and
distance to the recent 20-day support/resistance range.

Definition (simplified, backward-looking only — no lookahead):
- resistance/support proxy = rolling 20-day high/low.
- "higher high" = current 20d rolling high > the 20d rolling high from
  20 days ago (i.e. the swing high made a new high vs the prior swing
  window). Same logic mirrored for higher low / lower high / lower low.

Comparisons against NaN (first ~40 days of a ticker's history, before two
full 20-day windows exist) are kept as NaN rather than silently resolving
to False, so "unknown" isn't confused with "not a higher high".
"""
import numpy as np
import pandas as pd


def _tri_state_compare(current: pd.Series, previous: pd.Series, op) -> pd.Series:
    result = op(current, previous).astype(float)
    return result.where(previous.notna() & current.notna())


def compute_structure(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()
    prev_high = rolling_high.shift(window)
    prev_low = rolling_low.shift(window)

    out["higher_high_20d"] = _tri_state_compare(rolling_high, prev_high, np.greater)
    out["lower_high_20d"] = _tri_state_compare(rolling_high, prev_high, np.less)
    out["higher_low_20d"] = _tri_state_compare(rolling_low, prev_low, np.greater)
    out["lower_low_20d"] = _tri_state_compare(rolling_low, prev_low, np.less)

    out["distance_to_resistance_pct"] = (rolling_high - close) / close * 100
    out["distance_to_support_pct"] = (close - rolling_low) / close * 100
    return out
