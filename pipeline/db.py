import datetime as dt
import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

load_dotenv()

metadata = MetaData()

# SQLite only auto-populates a PRIMARY KEY column on INSERT when its
# declared type compiles to the literal "INTEGER" (its ROWID-alias rule --
# BIGINT, even though same-range on SQLite, does NOT qualify and left `id`
# NULL on every insert, confirmed via a real ingestion run: "NOT NULL
# constraint failed: price_history.id"). with_variant keeps every other
# backend (MySQL/production) on a real BIGINT DDL, unchanged.
_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")

stocks = Table(
    "stocks",
    metadata,
    Column("code", String(10), primary_key=True),
    Column("name", String(255)),
    Column("sector", String(100)),
    Column("industry", String(100)),
    Column("created_at", DateTime, server_default=func.now()),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)

price_history = Table(
    "price_history",
    metadata,
    Column("id", _ID_TYPE, primary_key=True, autoincrement=True),
    Column("stock_code", String(10), nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Numeric(14, 2)),
    Column("high", Numeric(14, 2)),
    Column("low", Numeric(14, 2)),
    Column("close", Numeric(14, 2)),
    Column("volume", BigInteger),
    Column("dividends", Numeric(14, 4), default=0),
    Column("stock_splits", Numeric(10, 4), default=0),
    Column("source_provider", String(20), nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("stock_code", "date", "source_provider", name="uq_price_history_code_date_source"),
)


# Two backends, chosen by whether MYSQL_HOST is set:
# - Local/dev (this Windows checkout): no MYSQL_HOST -> a local SQLite file,
#   zero install/service, which is the whole point of dropping Docker for
#   local use. Overridable via MYSTOCKS_DB_PATH; relative paths resolve
#   from the repo root, not the caller's cwd, so this behaves the same
#   however a script is invoked.
# - Production (VPS, Fase 6): docker-compose.yml sets MYSQL_HOST=mysql (the
#   compose service name) -> unchanged MySQL connection, same as before
#   this file supported SQLite at all. Never both at once, so `get_engine`
#   callers don't need to know or care which one they got.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "data", "mystocks.db")


def get_engine():
    mysql_host = os.getenv("MYSQL_HOST")
    if mysql_host:
        port = os.getenv("MYSQL_PORT", "3306")
        database = os.getenv("MYSQL_DATABASE", "mystocks")
        user = os.getenv("MYSQL_USER", "")
        password = os.getenv("MYSQL_PASSWORD", "")
        url = f"mysql+pymysql://{user}:{password}@{mysql_host}:{port}/{database}"
        return create_engine(url, pool_pre_ping=True)

    db_path = os.getenv("MYSTOCKS_DB_PATH", DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    url = f"sqlite:///{db_path}"
    # check_same_thread=False: Streamlit and the scheduler each hand out
    # connections from this engine's pool across multiple threads (a fresh
    # DBAPI connection per checkout, never shared concurrently) -- SQLite's
    # default same-thread restriction is a blanket check on the raw
    # connection object, not an actual guard against real concurrent use,
    # and SQLAlchemy's pooling already keeps checkouts thread-safe.
    return create_engine(url, connect_args={"check_same_thread": False}, pool_pre_ping=True)


def init_schema(engine):
    metadata.create_all(engine)


def coerce_date(value):
    """A raw text()-wrapped query bypasses SQLAlchemy's normal Column-type
    result processing, which is where a DATE column's string form would
    otherwise get parsed back into a real date automatically. MySQL's
    pymysql driver happens to hand back a native datetime.date regardless
    (driver-level, not SQLAlchemy), so this was never exercised there --
    but SQLite's stdlib sqlite3 driver returns the stored value exactly as
    written, a plain ISO string, which then fails a `date > str` comparison
    the moment one is attempted (found for real: features/build_features.py's
    incremental-update check). Callers use this on any date/None value
    fetched via `text(...)` rather than a Table/Column select."""
    if value is None or isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value)


# A single SQLite statement caps its total bound parameters (999 pre-3.32,
# raised to 32766 since -- staying under the OLDER number keeps this
# correct on any SQLite build, not just whatever this machine happens to
# ship, since Python's stdlib sqlite3 links whatever SQLite the interpreter
# was built against). A wide table's multi-row upsert can blow past that
# in one INSERT (e.g. feature_daily: ~48 cols x a ticker's full history in
# one call -- confirmed for real: "too many SQL variables" on a 1203-row
# batch), so rows are chunked to stay under it; harmless on MySQL too (no
# such limit there, just a few extra round trips for a very large batch).
SQLITE_MAX_VARIABLES = 900


def upsert(conn, table, rows: list[dict], update_columns: list[str], index_elements: list[str]) -> None:
    """Shared insert-or-update-on-conflict for every table in this project
    keyed by a UniqueConstraint (not the autoincrement `id` PK) -- e.g.
    price_history keyed on (stock_code, date, source_provider). Dialect is
    read off the live connection (not assumed) so every call site works
    unchanged against either backend get_engine() might have returned.
    SQLite needs the conflicting unique columns named explicitly via
    `index_elements` (unlike MySQL's ON DUPLICATE KEY UPDATE, which infers
    whichever unique index was violated) -- pass the same columns listed in
    that table's UniqueConstraint in db.py/features/db.py/engine/db.py;
    harmlessly unused on the MySQL path. No-ops on an empty `rows` list
    rather than erroring on an empty VALUES clause."""
    if not rows:
        return
    is_mysql = conn.engine.dialect.name == "mysql"
    chunk_size = max(1, SQLITE_MAX_VARIABLES // len(rows[0]))
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        if is_mysql:
            stmt = mysql_insert(table).values(chunk)
            set_ = {c: stmt.inserted[c] for c in update_columns}
            stmt = stmt.on_duplicate_key_update(**set_)
        else:
            stmt = sqlite_insert(table).values(chunk)
            set_ = {c: stmt.excluded[c] for c in update_columns}
            stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=set_)
        conn.execute(stmt)
