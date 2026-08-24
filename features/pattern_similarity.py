"""Historical pattern similarity (within-ticker, walk-forward safe).

For each day t, take the trailing `window`-day cumulative-return shape of the
price and compare it (Pearson correlation of the normalized shape) against
every PRIOR non-overlapping window of the same ticker. Two correctness
constraints that are easy to get wrong and matter a lot here:

1. No overlap: a candidate window ending at t' is only compared if
   t' <= t - window. Adjacent/overlapping windows share most of their days
   and would trivially score near-perfect similarity, making the whole
   feature meaningless.
2. No lookahead: a candidate window's forward outcome (used for
   historical_win_rate) is only used if it was already knowable as of day t,
   i.e. t' + horizon <= t. Otherwise we'd be leaking future information into
   a feature meant to describe what was known "at the time".

Implementation note: correlation between two z-scored vectors of length N
equals their dot product / N, so all pairwise correlations for a ticker are
computed as one matrix multiply (z @ z.T / window) rather than a Python
double loop — this keeps a ~1200-day history tractable (validated: <1s for
5 years of daily data in testing).
"""
import numpy as np
import pandas as pd

WINDOW = 20
HORIZON = 10
CORR_THRESHOLD = 0.85


def _build_normalized_windows(close: np.ndarray, window: int) -> np.ndarray:
    t_count = len(close)
    windows = np.full((t_count, window), np.nan)
    for t in range(window - 1, t_count):
        segment = close[t - window + 1 : t + 1]
        start = segment[0]
        if start == 0 or np.isnan(start):
            continue
        windows[t] = (segment / start - 1) * 100
    return windows


def _zscore_rows(windows: np.ndarray):
    t_count = windows.shape[0]
    row_mean = np.full((t_count, 1), np.nan)
    row_std = np.full((t_count, 1), np.nan)

    # Rows before `window` days of history exist are entirely NaN by
    # construction (see _build_normalized_windows). Only reduce rows that
    # have at least one real value — nanmean/nanstd on an all-NaN row would
    # still correctly return NaN, but numpy raises "Mean of empty slice" for
    # it, which would drown out warnings that actually mean something.
    has_any_value = ~np.isnan(windows).all(axis=1)
    if has_any_value.any():
        row_mean[has_any_value, 0] = np.nanmean(windows[has_any_value], axis=1)
        row_std[has_any_value, 0] = np.nanstd(windows[has_any_value], axis=1)

    valid_row = (row_std[:, 0] > 1e-9) & ~np.isnan(row_std[:, 0])
    z = np.full_like(windows, np.nan)
    z[valid_row] = (windows[valid_row] - row_mean[valid_row]) / row_std[valid_row]
    return z, valid_row


def compute_pattern_similarity(
    df: pd.DataFrame,
    window: int = WINDOW,
    horizon: int = HORIZON,
    corr_threshold: float = CORR_THRESHOLD,
) -> pd.DataFrame:
    close_series = df["close"]
    close = close_series.to_numpy(dtype=float)
    t_count = len(close)

    windows = _build_normalized_windows(close, window)
    z, valid_row = _zscore_rows(windows)
    corr = z @ z.T / window  # NaN rows propagate to NaN correlations, which is what we want

    forward_return = (close_series.pct_change(horizon).shift(-horizon) * 100).to_numpy()

    similarity_score = np.full(t_count, np.nan)
    similar_count = np.zeros(t_count, dtype=float)
    win_rate = np.full(t_count, np.nan)

    for t in range(t_count):
        if not valid_row[t]:
            continue
        max_t_prime = min(t - window, t - horizon)  # no overlap AND outcome must be known by t
        if max_t_prime < window - 1:
            continue

        candidate_idx = np.arange(window - 1, max_t_prime + 1)
        candidate_idx = candidate_idx[valid_row[candidate_idx]]
        if len(candidate_idx) == 0:
            continue

        corrs = corr[t, candidate_idx]
        keep = ~np.isnan(corrs)
        corrs = corrs[keep]
        candidate_idx = candidate_idx[keep]
        if len(corrs) == 0:
            continue

        similarity_score[t] = np.max(corrs)
        matched = candidate_idx[corrs > corr_threshold]
        similar_count[t] = len(matched)

        if len(matched) > 0:
            outcomes = forward_return[matched]
            outcomes = outcomes[~np.isnan(outcomes)]
            if len(outcomes) > 0:
                win_rate[t] = float((outcomes > 0).mean())

    return pd.DataFrame(
        {
            "similarity_score": similarity_score,
            "similar_pattern_count": similar_count,
            "historical_win_rate": win_rate,
        },
        index=df.index,
    )
