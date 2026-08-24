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


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns open/high/low/close/volume, indexed by date ascending."""
    parts = [
        compute_trend(df),
        compute_momentum(df),
        compute_volume(df),
        compute_money_flow(df),
        compute_volatility(df),
    ]
    return pd.concat(parts, axis=1)
