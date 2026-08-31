"""Export feature_daily + price_history (high/low/close, for label
construction) to Parquet files for upload to Google Colab.

Colab can't reach a local/dev MySQL instance over the network, so Fase 3
research loads from this file rather than a live DB connection. Re-run this
whenever you want the Colab notebook to see fresh data.

Usage:
    python -m scripts.export_for_colab [output_path]
"""
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Integer, Numeric, String, text

from engine.decision import GOCAP_PRICE_FLOOR
from features.db import feature_daily
from pipeline.db import get_engine, price_history
from pipeline.logging_config import get_logger

logger = get_logger("scripts.export_for_colab")

DEFAULT_OUTPUT = "data/export_for_colab.parquet"

_PYARROW_TYPE_MAP = [
    (Numeric, pa.float64()),
    (Boolean, pa.bool_()),
    (BigInteger, pa.int64()),
    (Integer, pa.int64()),
    (DateTime, pa.timestamp("us")),
    (Date, pa.date32()),
    (String, pa.string()),
]


def _select_sql(table, columns: list[str]) -> str:
    """Build a SELECT that CASTs every DECIMAL column to DOUBLE. Without
    this, pymysql hands each DECIMAL value back as a Python Decimal object
    -- ~10x the memory of a float64 and much slower to construct."""
    select_list = [
        f"CAST({c.name} AS DOUBLE) AS {c.name}" if isinstance(c.type, Numeric) else c.name
        for c in table.columns if c.name in columns
    ]
    return f"SELECT {', '.join(select_list)} FROM {table.name}"


def _pyarrow_schema(table, columns: list[str]) -> pa.Schema:
    """Explicit schema (from the SQLAlchemy table's own column types), not
    per-chunk type inference. Needed because this export processes one
    ticker at a time to stay memory-safe (see `export` below) -- a ticker
    whose ENTIRE subset happens to be NULL for some column (e.g. trailing_pe
    for a persistently loss-making company) would otherwise infer that
    chunk's column as pyarrow `null` type, which then fails to append onto
    an already-written float64 column from an earlier, non-null chunk."""
    fields = []
    for c in table.columns:
        if c.name not in columns:
            continue
        pa_type = next((t for cls, t in _PYARROW_TYPE_MAP if isinstance(c.type, cls)), pa.string())
        fields.append(pa.field(c.name, pa_type))
    return pa.schema(fields)


def export(output_path: str = DEFAULT_OUTPUT) -> None:
    engine = get_engine()
    features_cols = [c.name for c in feature_daily.columns if c.name not in ("id", "created_at")]
    price_cols = ["stock_code", "date", "high", "low", "close"]
    # text(...) here (not a plain string) so SQLAlchemy translates the
    # :code placeholder to whatever paramstyle the active backend's DBAPI
    # actually needs -- pd.read_sql sends a plain string straight to the
    # driver unmodified (pyformat for pymysql/MySQL, but sqlite3 doesn't
    # understand pyformat at all), while a text() clause goes through
    # SQLAlchemy's own dialect-aware compiler first.
    features_sql = text(_select_sql(feature_daily, features_cols) + " WHERE stock_code = :code")
    prices_sql = text(
        _select_sql(price_history, price_cols) + " WHERE stock_code = :code AND source_provider='yfinance'"
    )
    features_schema = _pyarrow_schema(feature_daily, features_cols)
    prices_schema = _pyarrow_schema(price_history, price_cols)

    with engine.connect() as conn:
        codes = [r[0] for r in conn.execute(text("SELECT DISTINCT stock_code FROM feature_daily")).fetchall()]

    # Processed ONE TICKER AT A TIME and streamed straight to Parquet
    # (ParquetWriter, not pandas.to_parquet on one big DataFrame) -- pulling
    # all ~1M feature_daily rows into a single DataFrame at once was enough
    # to get OOM-killed in this sandbox's constrained memory. Per-ticker
    # keeps peak memory to ~1200 rows at a time, same batching philosophy
    # features.build_features already uses for the same reason.
    features_path = output_path.replace(".parquet", "_features.parquet")
    prices_path = output_path.replace(".parquet", "_prices.parquet")
    features_writer = pq.ParquetWriter(features_path, features_schema)
    prices_writer = pq.ParquetWriter(prices_path, prices_schema)

    total_features = 0
    total_gocap = 0
    total_prices = 0
    try:
        for i, code in enumerate(codes, 1):
            feat_df = pd.read_sql(features_sql, engine, params={"code": code})
            price_df = pd.read_sql(prices_sql, engine, params={"code": code})

            # Exclude gocap-floor rows as TRAINING EXAMPLES only --
            # `prices` stays untouched so other (non-gocap) rows' forward-
            # looking triple-barrier labels still see a complete,
            # continuous price series. See engine.decision.GOCAP_PRICE_FLOOR
            # for why these rows are mostly tick-size noise, not signal.
            merged = feat_df.merge(price_df[["stock_code", "date", "close"]], on=["stock_code", "date"], how="left")
            gocap_mask = merged["close"] <= GOCAP_PRICE_FLOOR
            total_gocap += int(gocap_mask.sum())
            feat_df = merged[~gocap_mask].drop(columns=["close"])

            features_writer.write_table(pa.Table.from_pandas(feat_df, schema=features_schema, preserve_index=False))
            prices_writer.write_table(pa.Table.from_pandas(price_df, schema=prices_schema, preserve_index=False))
            total_features += len(feat_df)
            total_prices += len(price_df)

            if i % 100 == 0 or i == len(codes):
                logger.info("[%d/%d] %s: %d feature rows, %d price rows so far", i, len(codes), code,
                            total_features, total_prices)
    finally:
        features_writer.close()
        prices_writer.close()

    logger.info(
        "Excluded %d feature_daily rows at/below the Rp%d gocap floor from the training export",
        total_gocap, GOCAP_PRICE_FLOOR,
    )
    logger.info(
        "Exported %d feature_daily rows and %d price_history rows to %s / %s",
        total_features, total_prices, features_path, prices_path,
    )


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
