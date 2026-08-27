"""Fase 2 orchestrator: reads price_history from MySQL, computes technical +
fundamental features, writes to feature_daily / feature_fundamental_snapshot.

Usage:
    python -m features.build_features
"""
import datetime as dt
import math

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from features.db import FEATURE_VERSION, feature_daily, feature_fundamental_snapshot, init_schema
from features.fundamental import IHSG_SYMBOL, compute_relative_strength, fetch_fundamental_snapshot
from features.pattern_similarity import compute_cross_ticker_pattern_similarity
from features.regime import classify_regime
from features.structure import compute_structure
from features.technical import MIN_ROWS_FOR_TECHNICAL_FEATURES
from features.technical import compute_all as compute_technical
from pipeline.db import get_engine, price_history
from pipeline.logging_config import get_logger
from pipeline.tickers import SEED_TICKERS, to_yfinance_symbol
from pipeline.yfinance_source import fetch_history

logger = get_logger("features.build_features")

BOOL_COLS = ("higher_high_20d", "higher_low_20d", "lower_high_20d", "lower_low_20d")
INT_COLS = ("obv", "similar_pattern_count")
PATTERN_QUERY_BATCH_SIZE = 150


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _safe_bool(value):
    if value is None or pd.isna(value):
        return None
    return bool(value)


def load_price_history(conn, code: str) -> pd.DataFrame:
    rows = conn.execute(
        select(
            price_history.c.date,
            price_history.c.open,
            price_history.c.high,
            price_history.c.low,
            price_history.c.close,
            price_history.c.volume,
        )
        .where(price_history.c.stock_code == code)
        .order_by(price_history.c.date.asc())
    ).fetchall()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df.set_index("date")
    df = df.set_index("date")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def build_technical_features(
    df: pd.DataFrame, index_close: pd.Series, pattern: pd.DataFrame, sector_close: pd.Series = None,
) -> pd.DataFrame:
    technical = compute_technical(df, index_close, sector_close)
    structure = compute_structure(df)
    merged = pd.concat([technical, structure], axis=1)
    merged["regime"] = classify_regime(merged, df["close"])
    pattern = pattern.set_axis(merged.index)  # pattern was computed on a reset-index copy of df
    return pd.concat([merged, pattern], axis=1)


def _build_sector_composites(price_by_code: dict, sector_by_code: dict) -> dict:
    """Equal-weight synthetic sector index per sector, built from whichever
    of `price_by_code`'s tickers belong to it. Composite is the cross-
    sectional MEAN of each member's daily % return (skipna), compounded into
    a level series -- not a "rebase everyone to 100 on one shared date"
    index, on purpose: with members listed at very different dates, a shared
    rebase date collapses to whenever the youngest member started trading,
    throwing away most of the sector's usable history. Averaging daily
    returns instead has no such floor -- a member just doesn't contribute on
    days it has no data yet, same idea as IHSG being used for
    relative_strength_20d_pct, but scoped to a ticker's own peer group
    instead of the whole market (see technical.compute_all's docstring for
    why that's a sharper signal).

    NOTE: on a partial run (e.g. the Home "Update harga" button, scoped to
    just the on-screen tickers) each sector's composite is only as
    representative as however many of that sector's members happen to be in
    this call's `tickers` -- fine for a live-price refresh, but a full run
    (all ~900 tickers) is what training data should come from.
    """
    sector_returns = {}
    for code, df in price_by_code.items():
        sector = sector_by_code.get(code)
        if sector:
            sector_returns.setdefault(sector, []).append(df["close"].pct_change())

    composites = {}
    for sector, returns_list in sector_returns.items():
        mean_daily_return = pd.concat(returns_list, axis=1).mean(axis=1, skipna=True)
        composites[sector] = (1 + mean_daily_return.fillna(0)).cumprod() * 100
    return composites


def _merge_fundamental_features(features: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    """Stamp trailing_pe / price_to_book / market_cap_log as a CONSTANT
    value across every row being written this call. feature_fundamental_
    snapshot only has a few days of history (recently added), nowhere near
    the ~3-year price_history window walk-forward training uses, so there is
    no true point-in-time fundamental series to join against past dates.
    Using today's snapshot as a static "what kind of company is this" proxy
    for a ticker's whole history is a deliberate approximation -- reasonable
    given these move slowly relative to a 10-day prediction horizon, but NOT
    a point-in-time backtest. Flagged here so that limitation doesn't get
    lost by the time this reaches training."""
    features = features.copy()
    features["trailing_pe"] = snapshot.get("trailing_pe")
    features["price_to_book"] = snapshot.get("price_to_book")
    market_cap = snapshot.get("market_cap")
    features["market_cap_log"] = math.log10(market_cap) if market_cap and market_cap > 0 else None
    return features


def upsert_feature_daily(conn, code: str, features: pd.DataFrame) -> int:
    if features.empty:
        return 0

    rows = []
    for date, row in features.iterrows():
        record = {"stock_code": code, "date": date, "feature_version": FEATURE_VERSION}
        for col in features.columns:
            value = row[col]
            if col in BOOL_COLS:
                record[col] = _safe_bool(value)
            elif col in INT_COLS:
                record[col] = _safe_int(value)
            elif col == "regime":
                record[col] = value if pd.notna(value) else None
            else:
                record[col] = _safe_float(value)
        rows.append(record)

    stmt = mysql_insert(feature_daily).values(rows)
    update_cols = {c: stmt.inserted[c] for c in features.columns}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)
    return len(rows)


def upsert_fundamental_snapshot(conn, code: str, snapshot: dict) -> None:
    row = {"stock_code": code, "snapshot_date": dt.date.today()}
    for key, value in snapshot.items():
        row[key] = _safe_int(value) if key == "market_cap" else _safe_float(value)

    stmt = mysql_insert(feature_fundamental_snapshot).values([row])
    update_cols = {c: stmt.inserted[c] for c in row if c not in ("stock_code", "snapshot_date")}
    stmt = stmt.on_duplicate_key_update(**update_cols)
    conn.execute(stmt)


def _load_ihsg_close(period: str = "max") -> pd.Series:
    try:
        ihsg_df = fetch_history(IHSG_SYMBOL, period=period)
    except Exception as exc:
        logger.warning("Failed to fetch IHSG (%s): %s — relative_strength will be null", IHSG_SYMBOL, exc)
        return pd.Series(dtype=float)

    if ihsg_df is None or ihsg_df.empty:
        return pd.Series(dtype=float)

    close = ihsg_df["Close"]
    close.index = close.index.date
    return close


def run(tickers=None):
    tickers = tickers or SEED_TICKERS
    engine = get_engine()
    init_schema(engine)

    # --- Load every qualifying ticker's price history once, up front. Reused
    # both for the cross-ticker pattern-similarity bank (needs everyone's
    # history at once) and the per-ticker technical/structure/regime pass
    # below -- avoids querying price_history twice per ticker.
    price_by_code = {}
    for code in tickers:
        with engine.connect() as conn:
            df = load_price_history(conn, code)
        if df.empty:
            logger.warning("%s: no price_history yet — run pipeline.ingest_price first, skipping", code)
            continue
        if len(df) < MIN_ROWS_FOR_TECHNICAL_FEATURES:
            logger.warning(
                "%s: only %d days of price_history (<%d minimum), skipping technical features for now",
                code, len(df), MIN_ROWS_FOR_TECHNICAL_FEATURES,
            )
            continue
        price_by_code[code] = df

    # This sandbox has been getting interrupted mid-run repeatedly (Codespace
    # idle-timeout killing the whole VM, not just this process -- see chat),
    # AND -- more importantly for routine daily runs -- re-querying a
    # ticker's entire ~1200-day history every day just to add one new row
    # is most of what made both the backfill AND a plain "update today's
    # price" run take hours. `last_feature_date` (per ticker) drives
    # `only_dates_after` in compute_cross_ticker_pattern_similarity, so a
    # ticker with nothing new to add costs next to nothing here, and one
    # with one new day only gets THAT day queried, not its whole history.
    # A ticker absent from this dict (brand new) gets its full history
    # queried, same as before. No adx_14-IS-NOT-NULL filter here on purpose:
    # the v3 schema migration backfill is long complete (measured: 33/993k
    # rows, all one ticker whose data genuinely can't produce an ADX value,
    # not stale pre-migration rows), and that filter defeated the index --
    # MySQL couldn't loose-scan the (stock_code, date, feature_version) index
    # for MAX(date) per group with a non-indexed WHERE column in the mix, so
    # it fell back to a ~979k-row full index scan (~20s) instead of ~900
    # index-only lookups (~0.1s) on every single call, including this
    # button's.
    with engine.connect() as conn:
        last_feature_date = dict(conn.execute(text(
            "SELECT stock_code, MAX(date) FROM feature_daily GROUP BY stock_code"
        )).fetchall())

    # Short-circuit before the expensive IHSG fetch (a "max"-period yfinance
    # call, ~25s regardless of scope) when every requested ticker is already
    # fully scored through its latest price_history row -- the common case
    # for the Home "Update harga" button, which re-runs this for the same
    # ~15 on-screen tickers and, outside of trading hours or on a repeat
    # click, usually has nothing new to compute. Measured: cuts a no-op call
    # from ~30s to under 1s.
    needs_update = [
        code for code, df in price_by_code.items()
        if code not in last_feature_date or df.index.max() > last_feature_date[code]
    ]
    if not needs_update:
        logger.info(
            "All %d requested tickers already scored through their latest price_history row, "
            "nothing to compute -- skipping IHSG fetch and pattern similarity",
            len(price_by_code),
        )
        return {"feature_daily_rows": 0, "fundamental_snapshots": 0, "failures": []}

    # Fetched once, full history: needed both for the live "latest" relative
    # strength (fundamental snapshot) and now also the historical per-day
    # version (feature_daily), which needs IHSG's value on every past date,
    # not just the last year.
    logger.info("Fetching IHSG (%s) full history for relative strength", IHSG_SYMBOL)
    ihsg_close = _load_ihsg_close()

    with engine.connect() as conn:
        sector_by_code = dict(conn.execute(text("SELECT code, sector FROM stocks")).fetchall())
    sector_composites = _build_sector_composites(price_by_code, sector_by_code)
    logger.info(
        "Built %d sector composites from %d tickers for sector_relative_strength_20d_pct",
        len(sector_composites), len(price_by_code),
    )

    # The bank (pass 1 inside compute_cross_ticker_pattern_similarity) is
    # cheap to rebuild (~10s for ~900 tickers); querying it per-ticker (pass
    # 2) is the slow, memory-sensitive part -- though with only_dates_after
    # above, most tickers on a routine run now query just one row, not
    # their whole history. Still batching + writing incrementally so a
    # crash partway through only costs the current batch, not the whole run.
    pattern_input = {code: df.reset_index() for code, df in price_by_code.items()}
    all_codes = list(price_by_code.keys())
    batches = [all_codes[i:i + PATTERN_QUERY_BATCH_SIZE] for i in range(0, len(all_codes), PATTERN_QUERY_BATCH_SIZE)]
    already_scored_count = sum(1 for c in all_codes if c in last_feature_date)
    logger.info(
        "Computing cross-ticker pattern similarity for %d tickers (%d already scored before, cheap "
        "incremental query; %d never scored, full history), in %d batches of up to %d",
        len(all_codes), already_scored_count, len(all_codes) - already_scored_count,
        len(batches), PATTERN_QUERY_BATCH_SIZE,
    )

    total_daily = 0
    total_fundamental = 0
    failures = []

    for batch_i, batch_codes in enumerate(batches):
        logger.info("Pattern batch %d/%d (%d tickers)...", batch_i + 1, len(batches), len(batch_codes))
        pattern_results = compute_cross_ticker_pattern_similarity(
            pattern_input, query_codes=batch_codes, only_dates_after=last_feature_date,
        )

        for code in batch_codes:
            df = price_by_code[code]
            try:
                with engine.begin() as conn:
                    sector_close = sector_composites.get(sector_by_code.get(code))
                    features = build_technical_features(df, ihsg_close, pattern_results[code], sector_close)
                    cutoff = last_feature_date.get(code)
                    if cutoff is not None:
                        # only_dates_after left every row up to `cutoff` with NaN/0
                        # pattern columns (never queried, by design) -- upserting
                        # those would blank out already-correct historical data,
                        # so only the genuinely new rows get written. `features.index`
                        # holds plain datetime.date (from price_history's DATE column,
                        # via load_price_history's set_index) -- compare against
                        # `cutoff` as-is (also a datetime.date from MySQL), not a
                        # pd.Timestamp, which raises TypeError against a bare date.
                        features = features[features.index > cutoff]
                    if features.empty:
                        logger.info("%s: no new trading days since %s, nothing to write", code, cutoff)
                        continue

                    # Fetched here (before the upsert, not after like before) so
                    # trailing_pe/price_to_book/market_cap_log can be merged into
                    # `features` -- same live yfinance call as before, just
                    # reordered, not an extra fetch.
                    snapshot = fetch_fundamental_snapshot(to_yfinance_symbol(code))
                    features = _merge_fundamental_features(features, snapshot)

                    n = upsert_feature_daily(conn, code, features)
                    total_daily += n
                    logger.info("%s: upserted %d feature_daily rows", code, n)

                    snapshot["relative_strength_20d_pct"] = compute_relative_strength(df["close"], ihsg_close)
                    upsert_fundamental_snapshot(conn, code, snapshot)
                    total_fundamental += 1
            except Exception as exc:
                logger.error("%s: feature build failed, skipping — %s", code, exc)
                failures.append(code)

    logger.info(
        "Done. feature_daily rows: %d, fundamental snapshots: %d. Failures: %s",
        total_daily, total_fundamental, failures or "none",
    )
    return {"feature_daily_rows": total_daily, "fundamental_snapshots": total_fundamental, "failures": failures}


if __name__ == "__main__":
    run()
