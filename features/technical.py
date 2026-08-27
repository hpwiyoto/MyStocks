"""Technical features computed purely from OHLCV (price_history).

Base indicators (RSI/MACD/Bollinger/ATR/CMF/MFI/OBV) come from the `ta`
library — verified against real BBCA.JK data to produce sane, in-range
values, and confirmed compatible with our pinned pandas/numpy (unlike
pandas-ta, whose `numba` dependency forces a numpy downgrade and which
produced a suspicious 0.0 first-value RSI in testing). Derivatives
(slope/acceleration/distance) are computed manually on top since those are
project-specific, not something a generic TA library provides.

`df` in every function here is expected to be a price_history slice for ONE
ticker, indexed by date ascending, with columns open/high/low/close/volume
(lowercase, matching pipeline.db.price_history column names).
"""
import pandas as pd
import ta

# The `ta` library's AverageTrueRange raises IndexError (not a graceful NaN)
# when given fewer than its `window` (14) rows — confirmed by testing down to
# n=1..13. This threshold gives comfortable margin above every window used
# here (ATR/RSI/MFI=14, BB=20, MACD signal~35) so nothing crashes on a
# freshly-listed ticker with a short price history.
MIN_ROWS_FOR_TECHNICAL_FEATURES = 60


def _slope(series: pd.Series, n: int) -> pd.Series:
    return (series - series.shift(n)) / n


def _pct_distance(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b) / b * 100


def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_200"] = close.rolling(200).mean()
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["ema20_slope_5d"] = _slope(out["ema_20"], 5)
    out["ema20_accel_5d"] = out["ema20_slope_5d"] - out["ema20_slope_5d"].shift(5)
    out["sma50_slope_10d"] = _slope(out["sma_50"], 10)
    out["price_vs_sma50_pct"] = _pct_distance(close, out["sma_50"])
    out["ema9_vs_ema20_pct"] = _pct_distance(out["ema_9"], out["ema_20"])
    return out


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["rsi_14"] = rsi
    out["rsi_slope_3d"] = _slope(rsi, 3)
    out["rsi_slope_5d"] = _slope(rsi, 5)
    out["rsi_distance_50"] = rsi - 50

    macd_ind = ta.trend.MACD(close)
    macd_hist = macd_ind.macd_diff()
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_hist
    out["macd_hist_slope_3d"] = _slope(macd_hist, 3)
    out["macd_hist_accel_3d"] = out["macd_hist_slope_3d"] - out["macd_hist_slope_3d"].shift(3)
    return out


def compute_volume(df: pd.DataFrame) -> pd.DataFrame:
    volume = df["volume"]
    out = pd.DataFrame(index=df.index)
    vol_avg_20 = volume.rolling(20).mean()
    out["rvol_20"] = volume / vol_avg_20
    out["volume_slope_5d"] = _slope(volume, 5)
    return out


def compute_money_flow(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    cmf = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    out["cmf_20"] = cmf
    out["cmf_slope_5d"] = _slope(cmf, 5)

    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out["obv"] = obv
    out["obv_slope_5d"] = _slope(obv, 5)
    # Raw OBV is a running cumulative sum -- its absolute magnitude scales with
    # how long the ticker has been listed and its typical volume, so it's
    # meaningless to compare across tickers and was left out of every model
    # through v4 for exactly that reason. z-scoring against its own trailing
    # 20-day mean/std turns it into "is money flowing in/out more than this
    # ticker's own recent norm right now" -- comparable across tickers, and
    # usable as a real training feature.
    obv_mean_20 = obv.rolling(20).mean()
    obv_std_20 = obv.rolling(20).std().replace(0, float("nan"))  # avoid /0 on a flat/no-volume stretch
    out["obv_zscore_20"] = (obv - obv_mean_20) / obv_std_20

    mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    out["mfi_14"] = mfi
    out["mfi_slope_5d"] = _slope(mfi, 5)
    return out


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_pct_14"] = atr / close * 100

    bb = ta.volatility.BollingerBands(close, window=20)
    bb_width_pct = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    out["bb_width_pct"] = bb_width_pct
    out["bb_width_change_5d"] = bb_width_pct - bb_width_pct.shift(5)
    return out


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 20-day VWAP -- daily bars only (no intraday ticks here), so
    this is the standard multi-day approximation: volume-weighted average of
    each day's typical price ((H+L+C)/3), not a single trading day's VWAP.
    price_vs_vwap20_pct follows the same "distance from a volume-weighted
    reference line" idea as price_vs_sma50_pct, but weighted by how much
    actually traded on each day instead of treating every day equally."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    out = pd.DataFrame(index=df.index)
    vwap_20 = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    out["price_vs_vwap20_pct"] = _pct_distance(df["close"], vwap_20)
    return out


def compute_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """ADX: trend STRENGTH (0-100), independent of direction -- complements
    RSI/regime, which capture direction but not how strong the trend is."""
    high, low, close = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)
    out["adx_14"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    return out


def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    prev_close = df["close"].shift(1)
    out["overnight_gap_pct"] = (df["open"] - prev_close) / prev_close * 100
    return out


def compute_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df.index)
    out["day_of_week"] = dates.dayofweek.astype(float)  # 0=Monday .. 4=Friday
    out["is_month_end_week"] = (dates.day >= 25).astype(float)
    return out


def compute_relative_strength_series(close: pd.Series, index_close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling version of features.fundamental.compute_relative_strength --
    one value per trading day (for training) instead of just the latest
    (that one stays as-is, used for the live UI's fundamental panel). NaN
    wherever `index_close` (IHSG) has no matching date."""
    if index_close is None or index_close.empty:
        return pd.Series(float("nan"), index=close.index)
    aligned_index = index_close.reindex(close.index)
    stock_return = close.pct_change(window)
    index_return = aligned_index.pct_change(window)
    return (stock_return - index_return) * 100


def compute_all(df: pd.DataFrame, index_close: pd.Series = None, sector_close: pd.Series = None) -> pd.DataFrame:
    """df must have columns open/high/low/close/volume, indexed by date
    ascending. `index_close` (optional): IHSG's own close series, date-
    indexed the same way, for relative_strength_20d_pct -- omit to leave
    that column NaN (e.g. when IHSG couldn't be fetched that run).
    `sector_close` (optional): this ticker's own sector's composite close
    series (see build_features._build_sector_composites), for
    sector_relative_strength_20d_pct -- a peer-group comparison, sharper
    than relative_strength_20d_pct's whole-market comparison since it
    controls for sector-wide moves (e.g. a broad banking rally lifting every
    bank at once isn't evidence any one of them has an edge over its peers)."""
    parts = [
        compute_trend(df),
        compute_momentum(df),
        compute_volume(df),
        compute_money_flow(df),
        compute_volatility(df),
        compute_vwap(df),
        compute_trend_strength(df),
        compute_gap(df),
        compute_calendar(df),
    ]
    result = pd.concat(parts, axis=1)
    result["relative_strength_20d_pct"] = compute_relative_strength_series(df["close"], index_close)
    result["sector_relative_strength_20d_pct"] = compute_relative_strength_series(df["close"], sector_close)
    return result
