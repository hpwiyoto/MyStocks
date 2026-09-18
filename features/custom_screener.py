"""Parameter registry + condition engine for the Screener Kustom page --
a Stockbit-style screener where the user picks WHICH columns to filter on
and their thresholds (or compares one column against another), instead of
the fixed set of dropdowns Momentum Screener/Swing offer. Kept separate
from app/pages so PARAM_REGISTRY and apply_conditions are testable without
a running Streamlit process.

Every key here must be an actual column on the frame app.data's
load_screener_universe() returns -- see that function's docstring for
where each one comes from (feature_daily, feature_fundamental_snapshot,
price_history, or app.data.load_liquidity/load_latest_predictions).
"""
import pandas as pd

# kind: "numeric" (full operator set, can compare to another numeric
# param), "boolean" (True/False only), "categorical" (isin one or more
# of the column's own distinct values, e.g. regime).
PARAM_REGISTRY: dict[str, dict] = {
    "close": {"label": "Harga (Close)", "category": "Harga & Likuiditas", "kind": "numeric"},
    "volume": {"label": "Volume", "category": "Harga & Likuiditas", "kind": "numeric"},
    "avg_traded_value": {"label": "Rata-rata Transaksi/hari (60h)", "category": "Harga & Likuiditas", "kind": "numeric"},
    "probability": {"label": "Probabilitas Swing (model aktif)", "category": "Harga & Likuiditas", "kind": "numeric"},

    "sma_20": {"label": "SMA 20", "category": "Tren", "kind": "numeric"},
    "sma_50": {"label": "SMA 50", "category": "Tren", "kind": "numeric"},
    "sma_200": {"label": "SMA 200", "category": "Tren", "kind": "numeric"},
    "ema_9": {"label": "EMA 9", "category": "Tren", "kind": "numeric"},
    "ema_20": {"label": "EMA 20", "category": "Tren", "kind": "numeric"},
    "ema_50": {"label": "EMA 50", "category": "Tren", "kind": "numeric"},
    "ema20_slope_5d": {"label": "Slope EMA20 (5 hari)", "category": "Tren", "kind": "numeric"},
    "ema20_accel_5d": {"label": "Akselerasi EMA20 (5 hari)", "category": "Tren", "kind": "numeric"},
    "sma50_slope_10d": {"label": "Slope SMA50 (10 hari)", "category": "Tren", "kind": "numeric"},
    "price_vs_sma50_pct": {"label": "Harga vs SMA50 (%)", "category": "Tren", "kind": "numeric"},
    "ema9_vs_ema20_pct": {"label": "EMA9 vs EMA20 (%)", "category": "Tren", "kind": "numeric"},

    "rsi_14": {"label": "RSI 14", "category": "Momentum", "kind": "numeric"},
    "rsi_slope_3d": {"label": "Slope RSI (3 hari)", "category": "Momentum", "kind": "numeric"},
    "rsi_slope_5d": {"label": "Slope RSI (5 hari)", "category": "Momentum", "kind": "numeric"},
    "rsi_distance_50": {"label": "Jarak RSI ke 50", "category": "Momentum", "kind": "numeric"},
    "macd": {"label": "MACD", "category": "Momentum", "kind": "numeric"},
    "macd_signal": {"label": "MACD Signal", "category": "Momentum", "kind": "numeric"},
    "macd_hist": {"label": "MACD Histogram", "category": "Momentum", "kind": "numeric"},
    "macd_hist_slope_3d": {"label": "Slope MACD Histogram (3 hari)", "category": "Momentum", "kind": "numeric"},
    "macd_hist_accel_3d": {"label": "Akselerasi MACD Histogram (3 hari)", "category": "Momentum", "kind": "numeric"},
    "adx_14": {"label": "ADX 14 (kekuatan tren)", "category": "Momentum", "kind": "numeric"},

    "rvol_20": {"label": "Volume Relatif (RVOL 20)", "category": "Volume & Money Flow", "kind": "numeric"},
    "volume_slope_5d": {"label": "Slope Volume (5 hari)", "category": "Volume & Money Flow", "kind": "numeric"},
    "cmf_20": {"label": "Chaikin Money Flow (20)", "category": "Volume & Money Flow", "kind": "numeric"},
    "cmf_slope_5d": {"label": "Slope CMF (5 hari)", "category": "Volume & Money Flow", "kind": "numeric"},
    "obv": {"label": "On-Balance Volume", "category": "Volume & Money Flow", "kind": "numeric"},
    "obv_slope_5d": {"label": "Slope OBV (5 hari)", "category": "Volume & Money Flow", "kind": "numeric"},
    "obv_zscore_20": {"label": "Z-score OBV (20)", "category": "Volume & Money Flow", "kind": "numeric"},
    "mfi_14": {"label": "Money Flow Index (14)", "category": "Volume & Money Flow", "kind": "numeric"},
    "mfi_slope_5d": {"label": "Slope MFI (5 hari)", "category": "Volume & Money Flow", "kind": "numeric"},
    "net_foreign_flow": {"label": "Net Foreign Flow (Rp)", "category": "Volume & Money Flow", "kind": "numeric"},

    "atr_pct_14": {"label": "ATR% (14)", "category": "Volatilitas", "kind": "numeric"},
    "bb_width_pct": {"label": "Lebar Bollinger Band (%)", "category": "Volatilitas", "kind": "numeric"},
    "bb_width_change_5d": {"label": "Perubahan Lebar BB (5 hari)", "category": "Volatilitas", "kind": "numeric"},

    "ret_10d_pct": {"label": "Return 10 hari (%)", "category": "Rally & VWAP", "kind": "numeric"},
    "ret_10d_atr_norm": {"label": "Return 10 hari (dinormalisasi ATR)", "category": "Rally & VWAP", "kind": "numeric"},
    "price_vs_vwap20_pct": {"label": "Harga vs VWAP20 (%)", "category": "Rally & VWAP", "kind": "numeric"},

    "distance_to_resistance_pct": {"label": "Jarak ke Resistance (%)", "category": "Struktur Pasar", "kind": "numeric"},
    "distance_to_support_pct": {"label": "Jarak ke Support (%)", "category": "Struktur Pasar", "kind": "numeric"},
    "higher_high_20d": {"label": "Higher High (20 hari)", "category": "Struktur Pasar", "kind": "boolean"},
    "higher_low_20d": {"label": "Higher Low (20 hari)", "category": "Struktur Pasar", "kind": "boolean"},
    "lower_high_20d": {"label": "Lower High (20 hari)", "category": "Struktur Pasar", "kind": "boolean"},
    "lower_low_20d": {"label": "Lower Low (20 hari)", "category": "Struktur Pasar", "kind": "boolean"},

    "regime": {"label": "Regime", "category": "Pola & Regime", "kind": "categorical"},
    "similarity_score": {"label": "Skor Kemiripan Pola Historis", "category": "Pola & Regime", "kind": "numeric"},
    "similar_pattern_count": {"label": "Jumlah Pola Historis Mirip", "category": "Pola & Regime", "kind": "numeric"},
    "historical_win_rate": {"label": "Win Rate Pola Historis", "category": "Pola & Regime", "kind": "numeric"},

    "overnight_gap_pct": {"label": "Gap Overnight (%)", "category": "Lainnya", "kind": "numeric"},
    "relative_strength_20d_pct": {"label": "Relative Strength vs IHSG (20h, %)", "category": "Lainnya", "kind": "numeric"},
    "sector_relative_strength_20d_pct": {"label": "Relative Strength vs Sektor (20h, %)", "category": "Lainnya", "kind": "numeric"},

    "trailing_pe": {"label": "Trailing PE", "category": "Fundamental", "kind": "numeric"},
    "price_to_book": {"label": "Price to Book", "category": "Fundamental", "kind": "numeric"},
    "market_cap_log": {"label": "Market Cap (log)", "category": "Fundamental", "kind": "numeric"},
    "forward_pe": {"label": "Forward PE", "category": "Fundamental", "kind": "numeric"},
    "peg_ratio": {"label": "PEG Ratio", "category": "Fundamental", "kind": "numeric"},
    "return_on_equity": {"label": "ROE", "category": "Fundamental", "kind": "numeric"},
    "return_on_assets": {"label": "ROA", "category": "Fundamental", "kind": "numeric"},
    "profit_margins": {"label": "Margin Laba", "category": "Fundamental", "kind": "numeric"},
    "operating_margins": {"label": "Margin Operasi", "category": "Fundamental", "kind": "numeric"},
    "dividend_yield": {"label": "Dividend Yield", "category": "Fundamental", "kind": "numeric"},
    "payout_ratio": {"label": "Payout Ratio", "category": "Fundamental", "kind": "numeric"},
    "market_cap": {"label": "Market Cap (Rp, mentah)", "category": "Fundamental", "kind": "numeric"},
    "held_percent_insiders": {"label": "% Kepemilikan Insider", "category": "Fundamental", "kind": "numeric"},
    "held_percent_institutions": {"label": "% Kepemilikan Institusi", "category": "Fundamental", "kind": "numeric"},
    "recommendation_mean": {"label": "Rata-rata Rekomendasi Analis", "category": "Fundamental", "kind": "numeric"},
    "target_mean_price": {"label": "Target Harga Analis (rata2)", "category": "Fundamental", "kind": "numeric"},
    "analyst_upside_pct": {"label": "Upside ke Target Analis (%)", "category": "Fundamental", "kind": "numeric"},
}

NUMERIC_PARAMS = [k for k, v in PARAM_REGISTRY.items() if v["kind"] == "numeric"]
CATEGORIES = sorted({v["category"] for v in PARAM_REGISTRY.values()})

OPERATORS = [">", "≥", "<", "≤", "=", "antara"]  # ≥/≤ = >=/<=


def param_options_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for key, meta in PARAM_REGISTRY.items():
        out[meta["category"]].append(key)
    return out


def apply_conditions(df: pd.DataFrame, conditions: list[dict]) -> pd.DataFrame:
    """conditions: list of dicts, always ANDed together (matching how a
    Stockbit-style screener stacks filters) -- shape per kind:
      numeric:     {"param", "kind": "numeric", "operator", "compare_to" (optional
                    other numeric param key) OR "value" (float, or (lo, hi) if
                    operator == "antara")}
      boolean:     {"param", "kind": "boolean", "value": True/False}
      categorical: {"param", "kind": "categorical", "value": [allowed values]}
    Unknown/missing param columns are skipped rather than raising -- lets
    the page stay resilient if a condition references a column this run's
    universe frame doesn't have (e.g. no fundamental snapshot ingested yet).
    """
    out = df
    for cond in conditions:
        key = cond["param"]
        if key not in out.columns:
            continue
        kind = cond["kind"]
        if kind == "boolean":
            col = out[key].fillna(False).astype(bool)
            out = out[col == bool(cond["value"])]
        elif kind == "categorical":
            allowed = cond["value"]
            if allowed:
                out = out[out[key].isin(allowed)]
        else:
            col = pd.to_numeric(out[key], errors="coerce")
            op = cond["operator"]
            if cond.get("compare_to"):
                other = pd.to_numeric(out[cond["compare_to"]], errors="coerce")
                out = out[_compare(col, op, other)]
            elif op == "antara":
                lo, hi = cond["value"]
                out = out[col.between(lo, hi)]
            else:
                out = out[_compare(col, op, cond["value"])]
    return out


def _compare(col: pd.Series, op: str, other) -> pd.Series:
    if op == ">":
        return col > other
    if op == "≥":
        return col >= other
    if op == "<":
        return col < other
    if op == "≤":
        return col <= other
    return col == other
