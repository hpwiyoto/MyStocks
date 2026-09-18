"""Schema for the app's own login/approval gate -- separate from
pipeline/db.py, features/db.py, and engine/db.py (each of those owns its
own MetaData + init_schema, the established pattern in this codebase; see
pipeline/db.py's docstring comments for the reasoning). This one is only
ever touched by the Streamlit app itself (app/auth.py), never by the CLI
data pipeline.
"""
from sqlalchemy import BigInteger, Column, Date, DateTime, Integer, MetaData, Numeric, String, Table, func, inspect, text

metadata = MetaData()

# See pipeline/db.py's _ID_TYPE comment -- SQLite only auto-populates a
# PRIMARY KEY on INSERT when its declared type is literally "INTEGER".
_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")

# status: "pending" (just signed in with Google for the first time, not
# decided yet) / "approved" (can use the app) / "rejected" (signed in but
# denied -- kept as a row, not deleted, so a repeat sign-in doesn't quietly
# re-queue them as a fresh "pending" request).
app_users = Table(
    "app_users",
    metadata,
    Column("email", String(255), primary_key=True),
    Column("name", String(255)),
    Column("status", String(20), nullable=False, default="pending"),
    Column("requested_at", DateTime, server_default=func.now()),
    Column("decided_at", DateTime),
)

# A user's own "I bought this" mark on a real BUY signal (direct user
# request: track a position's progress day by day after they mark it, and
# warn early if it's drifting toward the stop-loss instead of waiting for
# the full -X% to hit -- see engine/early_warning.py). Deliberately in
# app/db.py, not engine/db.py's `predictions` -- this is a per-user
# ACTION on top of a prediction, never written by the CLI pipeline, same
# reasoning as app_users above.
# status: "active" (still being tracked) / "closed" (hit target/stop, or
# the user closed it manually, or HORIZON days passed with no touch --
# see closed_reason for which).
tracked_positions = Table(
    "tracked_positions",
    metadata,
    Column("id", _ID_TYPE, primary_key=True, autoincrement=True),
    Column("user_email", String(255), nullable=False),
    Column("stock_code", String(10), nullable=False),
    Column("model_version", String(30), nullable=False),
    Column("entry_date", Date, nullable=False),
    Column("entry_price", Numeric(14, 2), nullable=False),
    Column("stop_loss_price", Numeric(14, 2), nullable=False),
    Column("take_profit_price", Numeric(14, 2), nullable=False),
    # Snapshot of what the Swing recommendation actually said AT THE
    # MOMENT this was marked -- direct user request ("direcord... hasil
    # rekomendasi swing nya apa saat di klik tandai beli"). Stored as a
    # snapshot, not looked up live from the model's current metadata,
    # because a config's OWN numbers can change later (the default config
    # gets retrained in place under the same model_version -- see scripts/
    # train_v5.py) -- without this, a position's recorded "5%/-2.5%/5
    # hari" could silently become wrong history if the model is retrained
    # again after this position was marked.
    Column("entry_probability", Numeric(6, 4)),
    Column("entry_regime", String(20)),
    Column("entry_wyckoff_phase", String(20)),
    Column("entry_target_pct", Numeric(6, 4)),
    Column("entry_stop_pct", Numeric(6, 4)),
    Column("entry_horizon_days", Integer),
    Column("status", String(12), nullable=False, default="active"),
    Column("closed_date", Date),
    Column("closed_price", Numeric(14, 2)),
    Column("closed_reason", String(20)),  # target_hit / stop_hit / early_warning / manual / expired
    # NOTE: target/stop-hit DATE is deliberately NOT a stored column here.
    # An earlier version of this feature persisted a "first detected"
    # timestamp (whenever someone happened to open Posisi Saya after the
    # crossing) -- direct user correction: that's the date the app was
    # OPENED, not the date it actually happened in the market. It's
    # computed live instead from price_history's daily high/low since
    # entry_date -- see app.positions._find_hit_dates -- which gives the
    # real historical date for free (already-stored daily OHLC) with no
    # schema/caching to keep in sync.
    Column("created_at", DateTime, server_default=func.now()),
)

# Columns added after tracked_positions first shipped -- metadata.create_
# all() below only creates MISSING tables, it never alters an existing
# one's columns, so a table created before this dict existed needs these
# added explicitly. ALTER TABLE ADD COLUMN is simple/portable enough
# (no default, no constraint) to work unchanged on both SQLite and MySQL.
_TRACKED_POSITIONS_ADDED_COLUMNS = {
    "entry_probability": "NUMERIC(6,4)",
    "entry_regime": "VARCHAR(20)",
    "entry_wyckoff_phase": "VARCHAR(20)",
    "entry_target_pct": "NUMERIC(6,4)",
    "entry_stop_pct": "NUMERIC(6,4)",
    "entry_horizon_days": "INTEGER",
}


def _migrate_tracked_positions(engine) -> None:
    inspector = inspect(engine)
    if "tracked_positions" not in inspector.get_table_names():
        return  # metadata.create_all() above already made a fully up-to-date table
    existing = {c["name"] for c in inspector.get_columns("tracked_positions")}
    missing = {c: t for c, t in _TRACKED_POSITIONS_ADDED_COLUMNS.items() if c not in existing}
    if not missing:
        return
    with engine.begin() as conn:
        for col, ddl_type in missing.items():
            conn.execute(text(f"ALTER TABLE tracked_positions ADD COLUMN {col} {ddl_type}"))


# Manually-maintained ticker exclusions that can't be detected from price
# data alone -- currently just the IDX Special Monitoring Board/FCA list
# (see scripts/special_monitoring_board.py's docstring for why: BEI's own
# site blocks automated scraping, so this started as a hardcoded Python
# file the user had to ask for a code change + app restart to refresh
# every time). Moved into the DB and editable from the Admin page instead
# so an update takes effect immediately, with no redeploy -- and
# "overwrite" (Admin's bulk-update button deletes then re-inserts) rather
# than an ever-growing history table, per direct user request to keep
# this lean.
manual_ticker_exclusion = Table(
    "manual_ticker_exclusion",
    metadata,
    Column("stock_code", String(10), primary_key=True),
    Column("reason", String(30), nullable=False, default="special_monitoring"),
    Column("note", String(255)),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)


def init_schema(engine):
    metadata.create_all(engine)
    _migrate_tracked_positions(engine)
