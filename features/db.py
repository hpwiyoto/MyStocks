from sqlalchemy import (
    BigInteger,
    Boolean,
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

FEATURE_VERSION = "v1"

feature_daily = Table(
    "feature_daily",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("stock_code", String(10), nullable=False),
    Column("date", Date, nullable=False),
    Column("feature_version", String(10), nullable=False),
    # Trend
    Column("sma_20", Numeric(14, 4)),
    Column("sma_50", Numeric(14, 4)),
    Column("sma_200", Numeric(14, 4)),
    Column("ema_9", Numeric(14, 4)),
    Column("ema_20", Numeric(14, 4)),
    Column("ema_50", Numeric(14, 4)),
    Column("ema20_slope_5d", Numeric(14, 6)),
    Column("ema20_accel_5d", Numeric(14, 6)),
    Column("sma50_slope_10d", Numeric(14, 6)),
    Column("price_vs_sma50_pct", Numeric(10, 4)),
    Column("ema9_vs_ema20_pct", Numeric(10, 4)),
    # Momentum
    Column("rsi_14", Numeric(8, 4)),
    Column("rsi_slope_3d", Numeric(10, 4)),
    Column("rsi_slope_5d", Numeric(10, 4)),
    Column("rsi_distance_50", Numeric(8, 4)),
    Column("macd", Numeric(14, 6)),
    Column("macd_signal", Numeric(14, 6)),
    Column("macd_hist", Numeric(14, 6)),
    Column("macd_hist_slope_3d", Numeric(14, 6)),
    Column("macd_hist_accel_3d", Numeric(14, 6)),
    # Volume
    Column("rvol_20", Numeric(10, 4)),
    Column("volume_slope_5d", Numeric(18, 4)),
    # Money flow
    Column("cmf_20", Numeric(8, 6)),
    Column("cmf_slope_5d", Numeric(10, 6)),
    Column("obv", BigInteger),
    Column("obv_slope_5d", Numeric(18, 4)),
    Column("mfi_14", Numeric(8, 4)),
    Column("mfi_slope_5d", Numeric(10, 4)),
    # Volatility
    Column("atr_pct_14", Numeric(8, 4)),
    Column("bb_width_pct", Numeric(10, 4)),
    Column("bb_width_change_5d", Numeric(10, 4)),
    # Market structure
    Column("higher_high_20d", Boolean),
    Column("higher_low_20d", Boolean),
    Column("lower_high_20d", Boolean),
    Column("lower_low_20d", Boolean),
    Column("distance_to_resistance_pct", Numeric(10, 4)),
    Column("distance_to_support_pct", Numeric(10, 4)),
    # Regime
    Column("regime", String(20)),
    # Historical pattern similarity
    Column("similarity_score", Numeric(6, 4)),
    Column("similar_pattern_count", Integer),
    Column("historical_win_rate", Numeric(6, 4)),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("stock_code", "date", "feature_version", name="uq_feature_daily_code_date_version"),
)

feature_fundamental_snapshot = Table(
    "feature_fundamental_snapshot",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("stock_code", String(10), nullable=False),
    Column("snapshot_date", Date, nullable=False),
    Column("trailing_pe", Numeric(10, 4)),
    Column("forward_pe", Numeric(10, 4)),
    Column("price_to_book", Numeric(10, 4)),
    Column("peg_ratio", Numeric(10, 4)),
    Column("return_on_equity", Numeric(10, 6)),
    Column("return_on_assets", Numeric(10, 6)),
    Column("profit_margins", Numeric(10, 6)),
    Column("operating_margins", Numeric(10, 6)),
    Column("dividend_yield", Numeric(8, 4)),
    Column("payout_ratio", Numeric(8, 4)),
    Column("market_cap", BigInteger),
    Column("held_percent_insiders", Numeric(8, 6)),
    Column("held_percent_institutions", Numeric(8, 6)),
    Column("recommendation_mean", Numeric(6, 4)),
    Column("target_mean_price", Numeric(14, 2)),
    Column("analyst_upside_pct", Numeric(10, 4)),
    Column("relative_strength_20d_pct", Numeric(10, 4)),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("stock_code", "snapshot_date", name="uq_feature_fundamental_code_date"),
)


def init_schema(engine):
    metadata.create_all(engine)
