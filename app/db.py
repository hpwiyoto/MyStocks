"""Schema for the app's own login/approval gate -- separate from
pipeline/db.py, features/db.py, and engine/db.py (each of those owns its
own MetaData + init_schema, the established pattern in this codebase; see
pipeline/db.py's docstring comments for the reasoning). This one is only
ever touched by the Streamlit app itself (app/auth.py), never by the CLI
data pipeline.
"""
from sqlalchemy import Column, DateTime, MetaData, String, Table, func

metadata = MetaData()

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


def init_schema(engine):
    metadata.create_all(engine)
