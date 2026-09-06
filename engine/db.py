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
    func,
)

metadata = MetaData()

# See pipeline/db.py's _ID_TYPE comment -- SQLite only auto-populates a
# PRIMARY KEY on INSERT when its declared type is literally "INTEGER".
_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")

predictions = Table(
    "predictions",
    metadata,
    Column("id", _ID_TYPE, primary_key=True, autoincrement=True),
    Column("stock_code", String(10), nullable=False),
    Column("date", Date, nullable=False),
    Column("model_version", String(30), nullable=False),
    Column("probability", Numeric(6, 4)),
    Column("decision", String(10)),  # BUY / WATCH / AVOID
    Column("entry_price", Numeric(14, 2)),
    Column("stop_loss_price", Numeric(14, 2)),
    Column("take_profit_price", Numeric(14, 2)),
    Column("risk_reward_ratio", Numeric(6, 2)),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("stock_code", "date", "model_version", name="uq_predictions_code_date_model"),
)


def init_schema(engine):
    metadata.create_all(engine)
