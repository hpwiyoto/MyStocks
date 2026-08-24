"""Rule-based market regime classifier (heuristic, not ML — Fase 2 explicitly
excludes model training). This is a documented first-pass definition meant to
be validated/refined empirically in Fase 3 (ML research), not a final truth.

Regimes, evaluated in priority order (first match wins):
1. overextended   — RSI > 70 and price far (>15%) above SMA50: likely due a pullback.
2. bearish        — price below a declining SMA50 which is itself below SMA200,
                     and RSI weak/falling: sustained downtrend.
3. bottoming      — price below SMA50 but RSI recovering from oversold (<40, rising):
                     downtrend losing steam.
4. early_reversal — price has just reclaimed EMA20 while EMA20 is still below
                     SMA50, and RSI is crossing up through 50: a nascent turn,
                     not yet a confirmed uptrend.
5. bullish        — price above a rising SMA50 which is above SMA200, RSI >= 50:
                     sustained uptrend.
6. accumulation   — volatility compressed (Bollinger width in the bottom 30% of
                     its own trailing 100-day range) with RSI drifting up and
                     price roughly flat vs SMA50: quiet build-up phase.
7. sideways       — fallback: none of the above conditions met.

Requires the merged output of features.technical.compute_all() (needs sma_50,
sma_200, rsi_14, rsi_slope_5d, price_vs_sma50_pct, bb_width_pct) plus the raw
`close` series.
"""
import numpy as np
import pandas as pd

BB_WIDTH_RANK_WINDOW = 100
ACCUMULATION_BB_RANK_THRESHOLD = 0.30
FLAT_PRICE_BAND_PCT = 5.0


def classify_regime(features: pd.DataFrame, close: pd.Series) -> pd.Series:
    sma_50 = features["sma_50"]
    sma_200 = features["sma_200"]
    rsi_14 = features["rsi_14"]
    rsi_slope_5d = features["rsi_slope_5d"]
    ema_20 = features["ema_20"]
    price_vs_sma50_pct = features["price_vs_sma50_pct"]
    bb_width_pct = features["bb_width_pct"]

    sma50_slope_positive = features["sma50_slope_10d"] > 0
    bb_rank = bb_width_pct.rolling(BB_WIDTH_RANK_WINDOW).rank(pct=True)

    overextended = (rsi_14 > 70) & (price_vs_sma50_pct > 15)
    bearish = (close < sma_50) & (sma_50 < sma_200) & (rsi_14 < 40) & (rsi_slope_5d <= 0)
    bottoming = (close < sma_50) & (rsi_14 < 40) & (rsi_slope_5d > 0)
    early_reversal = (close > ema_20) & (ema_20 <= sma_50) & (rsi_14.between(45, 65)) & (rsi_slope_5d > 0)
    bullish = (close > sma_50) & (sma_50 > sma_200) & sma50_slope_positive & (rsi_14 >= 50)
    accumulation = (
        (bb_rank < ACCUMULATION_BB_RANK_THRESHOLD)
        & (rsi_slope_5d > 0)
        & (price_vs_sma50_pct.abs() < FLAT_PRICE_BAND_PCT)
    )

    conditions = [overextended, bearish, bottoming, early_reversal, bullish, accumulation]
    choices = ["overextended", "bearish", "bottoming", "early_reversal", "bullish", "accumulation"]
    regime = pd.Series(np.select(conditions, choices, default="sideways"), index=features.index)

    # Rows where the inputs themselves are still NaN (warmup period, e.g. before
    # SMA200 has 200 days of history) shouldn't be labeled at all.
    required = [sma_50, sma_200, rsi_14, ema_20, bb_width_pct]
    has_data = pd.concat(required, axis=1).notna().all(axis=1)
    regime = regime.where(has_data)
    return regime
