"""Historical pattern similarity, walk-forward safe.

Two implementations:

- `compute_pattern_similarity` (legacy, within-ticker): compares a ticker's
  current pattern only against ITS OWN prior history. Kept for reference/
  tests. Coverage is weak for tickers with a short history -- 55.5% of
  feature_daily rows had zero matches (verified against the live DB,
  2026-08-25), since the search pool for any given ticker never grows no
  matter how many OTHER tickers get ingested.

- `compute_cross_ticker_pattern_similarity` (current, used by
  features.build_features): searches across ALL tracked tickers' history at
  once. A stock's setup today can match a historically similar setup from a
  completely different stock, not just its own past. This is what makes
  adding more tickers to the universe actually pay off for this feature.

For each day t, take the trailing `window`-day cumulative-return shape of the
price and compare it (Pearson correlation of the normalized shape) against
every eligible prior window (own-ticker or cross-ticker). Two correctness
constraints that are easy to get wrong and matter a lot here:

1. No overlap: a candidate window ending at t' is only compared if
   t' <= t - window. Adjacent/overlapping windows share most of their days
   and would trivially score near-perfect similarity, making the whole
   feature meaningless.
2. No lookahead: a candidate window's forward outcome (used for
   historical_win_rate) is only used if it was already knowable as of day t,
   i.e. t' + horizon <= t. Otherwise we'd be leaking future information into
   a feature meant to describe what was known "at the time".

The no-lookahead rule (constraint 2) has a useful side effect for the
cross-ticker version specifically: it also rules out contemporaneous matches
(e.g. two banking stocks that moved together during the same market-wide
rally) trivially inflating the match count, since any eligible candidate
must be resolved at least `horizon` trading days before the query date --
same-week matches from a different ticker are never eligible either. It does
NOT solve full sample-independence (several correlated tickers can still
each contribute a separate "match" for what was really one underlying market
event) -- accepted as a known limitation of a heuristic feature, not solved
here.

Implementation note: correlation between two z-scored vectors of length N
equals their dot product / N. Within one ticker this makes the full
similarity matrix a single matmul (z @ z.T / window). The cross-ticker
version does the same thing at bigger scale: query windows against the full
~1M-window bank in blocks (query_block @ bank.T / window), never
materializing the full N x N matrix at once. A `scipy.spatial.cKDTree`
radius query was tried first as a way to avoid the O(n_query * n_bank) cost
altogether, but rejected after measuring: at d=window=20 dimensions, KD-tree
radius queries degrade toward brute-force as the bank grows (the standard
curse-of-dimensionality failure mode for tree indexes) -- 12s for a
30-ticker bank vs 541s for 150 tickers, worse than quadratic. The blocked
matmul is still O(n_query * n_bank), but with a far smaller, predictable
constant factor (BLAS-backed, fully vectorized per block).
"""
import warnings

import numpy as np
import pandas as pd

from pipeline.logging_config import get_logger

logger = get_logger("features.pattern_similarity")

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


def compute_cross_ticker_pattern_similarity(
    price_by_ticker: dict,
    window: int = WINDOW,
    horizon: int = HORIZON,
    corr_threshold: float = CORR_THRESHOLD,
    query_codes=None,
    only_dates_after: dict = None,
) -> dict:
    """Cross-ticker version of the same feature. `price_by_ticker` maps stock
    code -> DataFrame with 'date' (ascending) and 'close' columns, index
    reset to 0..n-1. Returns {stock_code: DataFrame(similarity_score,
    similar_pattern_count, historical_win_rate)}, one row per input row.

    `query_codes` (optional): only compute results for these tickers, while
    still building the bank from -- and matching against -- every ticker in
    `price_by_ticker`. Lets a caller checkpoint a long run in batches (e.g.
    150 tickers' worth of results at a time, writing each batch to the DB
    before moving on) without losing cross-ticker correctness -- the bank
    itself is cheap to rebuild (~10s for ~900 tickers), it's the per-ticker
    querying that's slow and worth being able to resume partway through.

    `only_dates_after` (optional): {stock_code: date} -- for a ticker present
    here, only rows with date strictly after the given date are queried (the
    rest keep NaN/0 in the result, same as any other un-queried row). This is
    what makes a routine daily update cheap: re-querying a ticker's entire
    ~1200-day history every day to add ONE new row is most of what made the
    original backfill take hours, and none of yesterday's rows' results
    change by adding one more day of data -- their query set (every OTHER
    still-eligible window) is a superset that grows, but a ticker's already-
    computed row never needs revisiting. The bank itself still includes
    every ticker's full history regardless of this argument, since a new
    day's row from ANY other ticker is a legitimate match for it.

    Unlike the legacy version, `similarity_score` here is only ever the best
    match AMONG those already clearing `corr_threshold` (found via an exact
    radius query, see below) -- it's NaN exactly when similar_pattern_count
    is 0, never a sub-threshold "best effort" value. Simpler and arguably
    more honest: a similarity score with no threshold-clearing precedent
    behind it isn't a meaningful number to hand the model anyway.
    """
    codes = list(price_by_ticker.keys())

    # --- Pass 1: build one global bank of (z-scored window, metadata) across
    # every ticker's entire history, keeping only windows whose own 10-day
    # forward outcome is already defined (i.e. not still hanging off the end
    # of that ticker's history).
    per_ticker = {}
    bank_z, bank_ticker_id, bank_t_index, bank_resolved_date, bank_forward_return = [], [], [], [], []

    for ticker_id, code in enumerate(codes):
        df = price_by_ticker[code]
        close = df["close"].to_numpy(dtype=float)
        dates = pd.to_datetime(df["date"]).to_numpy()
        t_count = len(close)

        windows = _build_normalized_windows(close, window)
        z, valid_row = _zscore_rows(windows)
        forward_return = (df["close"].pct_change(horizon).shift(-horizon) * 100).to_numpy()
        resolved_date = np.full(t_count, np.datetime64("NaT"), dtype="datetime64[ns]")
        if t_count > horizon:
            resolved_date[: t_count - horizon] = dates[horizon:]

        per_ticker[code] = (valid_row, dates, z, resolved_date)

        bankable = valid_row & ~np.isnan(forward_return)
        n_bankable = int(bankable.sum())
        if n_bankable:
            idx = np.where(bankable)[0]
            bank_z.append(z[idx].astype(np.float32))
            bank_ticker_id.append(np.full(n_bankable, ticker_id, dtype=np.int32))
            bank_t_index.append(idx.astype(np.int32))
            bank_resolved_date.append(resolved_date[idx])
            bank_forward_return.append(forward_return[idx].astype(np.float32))

    if not bank_z:
        fallback_codes = codes if query_codes is None else query_codes
        return {code: pd.DataFrame({
            "similarity_score": np.nan, "similar_pattern_count": 0.0, "historical_win_rate": np.nan,
        }, index=price_by_ticker[code].index) for code in fallback_codes}

    bank_z = np.concatenate(bank_z, axis=0)
    bank_ticker_id = np.concatenate(bank_ticker_id)
    bank_t_index = np.concatenate(bank_t_index)
    bank_resolved_date = np.concatenate(bank_resolved_date)
    bank_forward_return = np.concatenate(bank_forward_return)
    logger.info("Pattern bank built: %d windows across %d tickers", len(bank_z), len(codes))

    # A cKDTree was tried first here and rejected: at d=window=20 dimensions,
    # radius queries degrade toward brute-force as the bank grows (the usual
    # curse-of-dimensionality failure mode for tree indexes) -- measured
    # 12s for a 30-ticker bank vs 541s for 150 tickers, i.e. worse than
    # quadratic, which would put the full ~900-ticker universe at several
    # hours. A blocked dense matmul is O(n_query * n_bank) too, but with a
    # far smaller constant factor (BLAS-backed, fully vectorized per block)
    # and predictable scaling, which is what actually matters at this size.
    bank_zT = bank_z.T  # (window, n_bank), reused across every block
    positive_outcome = (bank_forward_return > 0).astype(np.float32)  # (n_bank,)
    NAN32 = np.float32(np.nan)
    # ~6 temp arrays of shape (QUERY_BLOCK, n_bank) live per iteration; sized
    # to keep that comfortably under ~1GB total in this sandbox (n_bank here
    # is ~1M, each array ~1-4 bytes/element). QUERY_BLOCK=200 was tried first
    # and OOM-killed the process -- partly a real bug (np.where(eligible,
    # block_corr, np.nan) silently upcasts float32 -> float64, since plain
    # Python `nan` is a float64, roughly doubling that array), partly because
    # this sandbox's OTHER processes (editor, DB server) leave very little
    # headroom and that headroom isn't stable run to run -- QUERY_BLOCK=50
    # still OOM-killed on a run where ambient memory happened to be tighter.
    # Kept conservative here since the cost of a smaller block is just more
    # (cheap) Python-loop iterations, not more total work.
    QUERY_BLOCK = 10

    query_code_set = set(codes) if query_codes is None else set(query_codes)
    results = {}
    for ticker_id, code in enumerate(codes):
        if code not in query_code_set:
            continue
        if ticker_id % 100 == 0:
            logger.info("Cross-ticker similarity: %d/%d tickers queried", ticker_id, len(codes))
        valid_row, dates, z, _resolved_date = per_ticker[code]
        t_count = len(dates)
        similarity_score = np.full(t_count, np.nan)
        similar_count = np.zeros(t_count, dtype=float)
        win_rate = np.full(t_count, np.nan)

        query_idx = np.where(valid_row)[0]
        if only_dates_after and code in only_dates_after:
            cutoff = np.datetime64(only_dates_after[code])
            query_idx = query_idx[dates[query_idx] > cutoff]
        if len(query_idx) == 0:
            results[code] = pd.DataFrame({
                "similarity_score": similarity_score, "similar_pattern_count": similar_count,
                "historical_win_rate": win_rate,
            }, index=price_by_ticker[code].index)
            continue

        query_points = z[query_idx].astype(np.float32)
        query_dates = dates[query_idx]
        query_t_index = query_idx  # t_index within this ticker's own series

        for b_start in range(0, len(query_idx), QUERY_BLOCK):
            b_end = min(b_start + QUERY_BLOCK, len(query_idx))
            block_t = query_idx[b_start:b_end]
            block_corr = (query_points[b_start:b_end] @ bank_zT) / window  # (B, n_bank)

            resolved = bank_resolved_date[None, :] <= query_dates[b_start:b_end, None]
            same_ticker_overlap = (bank_ticker_id[None, :] == ticker_id) & (
                bank_t_index[None, :] > (query_t_index[b_start:b_end, None] - window)
            )
            eligible = resolved & ~same_ticker_overlap

            eligible_corr = np.where(eligible, block_corr, NAN32)
            with warnings.catch_warnings():
                # nanmax on an all-NaN row (no eligible candidate that day) correctly
                # returns NaN but warns every time -- expected here, not a real issue.
                warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                similarity_score[block_t] = np.nanmax(eligible_corr, axis=1)

            match_mask = eligible & (block_corr > corr_threshold)
            match_count = match_mask.sum(axis=1)
            similar_count[block_t] = match_count
            win_numerator = match_mask.astype(np.float32) @ positive_outcome
            with np.errstate(invalid="ignore", divide="ignore"):
                win_rate[block_t] = np.where(match_count > 0, win_numerator / np.maximum(match_count, 1), np.nan)

        results[code] = pd.DataFrame({
            "similarity_score": similarity_score,
            "similar_pattern_count": similar_count,
            "historical_win_rate": win_rate,
        }, index=price_by_ticker[code].index)

    return results
