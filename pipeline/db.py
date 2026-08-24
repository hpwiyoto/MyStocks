import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    MetaData,
    Numeric,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
)

load_dotenv()

metadata = MetaData()

stocks = Table(
    "stocks",
    metadata,
    Column("code", String(10), primary_key=True),
    Column("name", String(255)),
    Column("sector", String(100)),
    Column("created_at", DateTime, server_default=func.now()),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)

price_history = Table(
    "price_history",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
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


def get_engine():
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    database = os.getenv("MYSQL_DATABASE", "mystocks")
    user = os.getenv("MYSQL_USER", "")
    password = os.getenv("MYSQL_PASSWORD", "")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True)


def init_schema(engine):
    metadata.create_all(engine)
