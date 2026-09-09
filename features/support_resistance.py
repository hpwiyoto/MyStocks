"""Pivot-based support/resistance: swing highs/lows clustered into "zones"
price has touched multiple times -- a closer approximation of how a
discretionary trader (Ellen May, William Hartanto, etc.) actually reads
support/resistance than features/structure.py's rolling-20-day-high/low
proxy (which stays unchanged as the MODEL feature -- distance_to_
resistance_pct/distance_to_support_pct -- since it's already validated
there; this module is for (a) the Detail Saham chart's on-screen S/R
lines and (b) testing, via scripts/test_pivot_support_resistance.py,
whether a multi-touch zone is a MORE actionable Momentum Screener
criterion than that simple proxy).

No-lookahead: a swing high/low at index i is only "confirmed" once
`window` bars AFTER i exist -- it takes that long to know i really was a
local extreme, not just the most recent bar so far. A walk-forward/as-of
caller must only use pivots at index <= as_of_idx - window; every
function here that has an "as of" notion (find_swing_highs/lows already
require a full trailing window on both sides to emit an index at all) is
written so a caller iterating date-by-date naturally respects this: an
index only appears in the output once it's genuinely confirmable.
"""
import numpy as np

DEFAULT_WINDOW = 5        # bars each side a pivot must beat to count as a swing high/low
DEFAULT_MIN_GAP = 5       # trading days -- collapse pivots from the same swing into one
DEFAULT_TOLERANCE_PCT = 1.5  # pivots within this %% of each other cluster into one zone
DEFAULT_MAX_LOOKBACK = 250   # trading days -- roughly a year, long enough for real S/R, not ancient history
DEFAULT_MAX_DISTANCE_PCT = 15  # ignore a level this far from current price -- not "relevant" anymore


def _find_swing_points(values: np.ndarray, is_high: bool, window: int, min_gap: int) -> list[int]:
    """Shared logic behind find_swing_highs/find_swing_lows -- indices
    where `values` is the max (is_high=True) or min (is_high=False) within
    its own +/-window neighborhood, deduped so two candidates from the
    same swing (closer than min_gap trading days apart) collapse into
    whichever is the more extreme one. Same dedup approach as
    features.momentum_screener._find_swing_lows, generalized to both
    directions so this module has one implementation, not two copies."""
    n = len(values)
    cmp = (lambda i: values[i] == values[i - window: i + window + 1].max()) if is_high else \
          (lambda i: values[i] == values[i - window: i + window + 1].min())
    better = (lambda a, b: values[a] > values[b]) if is_high else (lambda a, b: values[a] < values[b])
    candidates = [i for i in range(window, n - window) if cmp(i)]
    points: list[int] = []
    for i in candidates:
        if not points or i - points[-1] >= min_gap:
            points.append(i)
        elif better(i, points[-1]):
            points[-1] = i
    return points


def find_swing_highs(high: np.ndarray, window: int = DEFAULT_WINDOW, min_gap: int = DEFAULT_MIN_GAP) -> list[int]:
    return _find_swing_points(high, is_high=True, window=window, min_gap=min_gap)


def find_swing_lows(low: np.ndarray, window: int = DEFAULT_WINDOW, min_gap: int = DEFAULT_MIN_GAP) -> list[int]:
    return _find_swing_points(low, is_high=False, window=window, min_gap=min_gap)


def cluster_levels(prices: list[float], indices: list[int], tolerance_pct: float = DEFAULT_TOLERANCE_PCT) -> list[dict]:
    """Groups pivot prices within tolerance_pct of each other into zones
    ("multi-touch" support/resistance). Greedy single pass over prices
    sorted ascending: a pivot joins the current cluster if it's within
    tolerance_pct of that cluster's running mean, else starts a new one --
    good enough for this use (pivots are already sparse after dedup
    above), not claiming globally-optimal clustering.

    Returns one dict per cluster: level (mean price), touches (pivot
    count), last_index (most recent pivot's index, for a recency
    tie-break) -- sorted by touches descending, then most recent first.
    """
    if not indices:
        return []
    pairs = sorted(zip(prices, indices), key=lambda pi: pi[0])
    clusters: list[dict] = []
    for price, idx in pairs:
        if clusters and abs(price - clusters[-1]["_mean"]) / clusters[-1]["_mean"] * 100 <= tolerance_pct:
            c = clusters[-1]
            c["_sum"] += price
            c["touches"] += 1
            c["_mean"] = c["_sum"] / c["touches"]
            c["last_index"] = max(c["last_index"], idx)
        else:
            clusters.append({"_sum": price, "touches": 1, "_mean": price, "last_index": idx})
    out = [{"level": c["_mean"], "touches": c["touches"], "last_index": c["last_index"]} for c in clusters]
    out.sort(key=lambda c: (-c["touches"], -c["last_index"]))
    return out


def compute_pivot_levels(
    high: np.ndarray, low: np.ndarray,
    window: int = DEFAULT_WINDOW, min_gap: int = DEFAULT_MIN_GAP,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT, max_lookback: int = DEFAULT_MAX_LOOKBACK,
) -> tuple[list[dict], list[dict]]:
    """Full-array version for the Detail Saham chart / any "as of the very
    last bar" use -- restricts to the last `max_lookback` bars first (a
    pivot from 3 years ago isn't "support" anymore), then clusters.
    Returns (support_clusters, resistance_clusters) -- support from swing
    LOWS, resistance from swing HIGHS, each cluster_levels()'s output."""
    n = len(high)
    start = max(0, n - max_lookback)
    high_w, low_w = high[start:], low[start:]
    high_idx = find_swing_highs(high_w, window, min_gap)
    low_idx = find_swing_lows(low_w, window, min_gap)
    resistance = cluster_levels([high_w[i] for i in high_idx], high_idx, tolerance_pct)
    support = cluster_levels([low_w[i] for i in low_idx], low_idx, tolerance_pct)
    return support, resistance


def nearest_significant_level(
    clusters: list[dict], current_price: float, side: str, max_distance_pct: float = DEFAULT_MAX_DISTANCE_PCT,
) -> dict | None:
    """Picks the most relevant level on one side of current_price: support
    (side="support") must be BELOW current_price, resistance ("resistance")
    must be ABOVE, both within max_distance_pct (a level 40% away isn't
    "the" support/resistance right now). Ranked by touches descending,
    then distance ascending -- a well-confirmed level a bit farther away
    still trumps a single-touch blip that happens to be closer, but among
    equally-touched levels the nearer one wins."""
    if not clusters or not current_price:
        return None
    candidates = []
    for c in clusters:
        if side == "support" and c["level"] >= current_price:
            continue
        if side == "resistance" and c["level"] <= current_price:
            continue
        distance_pct = abs(current_price - c["level"]) / current_price * 100
        if distance_pct > max_distance_pct:
            continue
        candidates.append({**c, "distance_pct": distance_pct})
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c["touches"], c["distance_pct"]))
    return candidates[0]
