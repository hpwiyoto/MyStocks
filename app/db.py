"""Schema for the app's own login/approval gate -- separate from
pipeline/db.py, features/db.py, and engine/db.py (each of those owns its
own MetaData + init_schema, the established pattern in this codebase; see
pipeline/db.py's docstring comments for the reasoning). This one is only
ever touched by the Streamlit app itself (app/auth.py), never by the CLI
data pipeline.
"""
from sqlalchemy import BigInteger, Column, Date, DateTime, Integer, MetaData, Numeric, String, Table, func

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
    Column("status", String(12), nullable=False, default="active"),
    Column("closed_date", Date),
    Column("closed_price", Numeric(14, 2)),
    Column("closed_reason", String(20)),  # target_hit / stop_hit / early_warning / manual / expired
    Column("created_at", DateTime, server_default=func.now()),
)


def init_schema(engine):
    metadata.create_all(engine)
