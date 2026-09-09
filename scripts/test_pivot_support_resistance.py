"""Does pivot-based, multi-touch support/resistance (features/
support_resistance.py) make a BETTER Momentum Screener criterion than the
already-tested "distance to rolling 50-day low" proxy from
scripts/test_strategy_6_criteria.py (dist_to_low50_pct <= 2%, which scored
33.6% alone -- weak, at/near the null baseline)?

Same no-lookahead 5-year historical dataset and walk-forward as-of-date
methodology as every other Momentum Screener backtest here
(scripts/search_momentum_rules.py's load_full_panel/triple_barrier_outcome/
wilson_lower_bound, reused directly). For each (ticker, as-of date) pair,
pivot levels are computed from ONLY the data up to and including that
as-of index -- features.support_resistance.find_swing_points structurally
cannot detect a pivot within its own trailing `window` bars of whatever
array it's given, so truncating the input array at the as-of index is
sufficient for no-lookahead correctness, no extra bookkeeping needed.

Candidates tested:
  - Standalone: close within N%% above ANY nearest pivot support.
  - Standalone: close within N%% above a MULTI-TOUCH (>=2 touches) pivot
    support specifically -- the whole point of building this over the
    simpler rolling-low proxy: does "touched more than once" carry more
    signal than "just happens to be a recent low"?
  - Standalone: enough room below a pivot resistance (>=N%% away) --
    Ellen-May-style "don't buy right under a ceiling."
  - Each layered on top of the already-shipped rule (bottoming+momentum+
    CMF<0+RVOL>=0.8), the same way Anchored VWAP was tested and adopted.

Usage:
    python -m scripts.test_pivot_support_resistance
"""
import numpy as np
import pandas as pd

from features.momentum_screener import (
    DEFAULT_REGIME_PRIORITY,
    REGIME_PRIORITY,
    classify_macd_status,
    detect_bullish_divergence,
)
from features.support_resistance import compute_pivot_levels, nearest_significant_level
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import (
    AS_OF_STRIDE,
    HORIZON,
    LOOKBACK_DAYS,
    WARMUP_DATES,
    load_full_panel,
    triple_barrier_outcome,
    wilson_lower_bound,
)

logger = get_logger("scripts.test_pivot_support_resistance")

SUPPORT_PROXIMITY_PCT = 2.0     # "close enough above support" bar, same threshold the 50d-low proxy used
RESISTANCE_ROOM_PCT = 5.0       # "enough room before resistance" bar


def build_dataset() -> pd.DataFrame:
    panel = load_full_panel()
    ticker_frames = {code: g.sort_values("date").reset_index(drop=True) for code, g in panel.groupby("stock_code")}
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates = sorted(panel["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates", len(as_of_dates))

    rows = []
    for n_done, as_of in enumerate(as_of_dates):
        for code, g in ticker_frames.items():
            idx = as_of_idx_by_ticker[code].get(as_of)
            if idx is None or idx < 30:
                continue
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window_df = g.iloc[start:idx + 1]
            latest = window_df.iloc[-1]
            if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]):
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue

            # Pivot levels from ONLY data through idx (inclusive) --
            # truncating here is what makes this no-lookahead, see module
            # docstring.
            high_asof = g["high"].to_numpy(dtype=float)[:idx + 1]
            low_asof = g["low"].to_numpy(dtype=float)[:idx + 1]
            support, resistance = compute_pivot_levels(high_asof, low_asof)
            close_now = float(latest["close"])
            nearest_sup = nearest_significant_level(support, close_now, "support")
            nearest_res = nearest_significant_level(resistance, close_now, "resistance")

            div = detect_bullish_divergence(window_df)
            rows.append({
                "as_of": as_of, "stock_code": code, "close": close_now,
                "rsi_14": latest["rsi_14"], "macd_status": classify_macd_status(window_df["macd_hist"]),
                "macd_hist_slope_3d": latest["macd_hist_slope_3d"],
                "cmf_20": latest["cmf_20"], "rvol_20": latest["rvol_20"], "regime": latest["regime"],
                "dist_to_pivot_support_pct": nearest_sup["distance_pct"] if nearest_sup else np.nan,
                "support_touches": nearest_sup["touches"] if nearest_sup else 0,
                "dist_to_pivot_resistance_pct": nearest_res["distance_pct"] if nearest_res else np.nan,
                "resistance_touches": nearest_res["touches"] if nearest_res else 0,
                "outcome": outcome, **div,
            })
        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rows so far)", n_done + 1, len(as_of_dates), len(rows))

    out = pd.DataFrame(rows)
    out["divergence_tier"] = np.select(
        [out["divergence_rsi"] & out["divergence_macd"], out["divergence_rsi"] | out["divergence_macd"]],
        [0, 1], default=2,
    )
    out["regime_priority"] = out["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
    logger.info("Dataset ready: %d resolved (ticker, as-of-date) rows", len(out))
    return out


def evaluate(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df[mask]
    n = len(sub)
    wins = int(sub["outcome"].sum())
    wr = wins / n if n else float("nan")
    return {"label": label, "n": n, "win_rate": wr, "wilson_lb": wilson_lower_bound(wins, n) if n else 0.0}


def run():
    df = build_dataset()
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(df), null_rate)
    logger.info("=" * 70)

    has_support = df["dist_to_pivot_support_pct"].notna()
    has_resistance = df["dist_to_pivot_resistance_pct"].notna()
    near_support = has_support & (df["dist_to_pivot_support_pct"] <= SUPPORT_PROXIMITY_PCT)
    near_support_multi = near_support & (df["support_touches"] >= 2)
    room_to_resistance = has_resistance & (df["dist_to_pivot_resistance_pct"] >= RESISTANCE_ROOM_PCT)

    shipped = (
        (df["regime"] == "bottoming")
        & (df["macd_hist_slope_3d"] > 0)
        & (df["cmf_20"] < 0)
        & (df["rvol_20"] >= 0.8)
    )

    candidates = [
        ("NULL (reference row, same as baseline above)", pd.Series(True, index=df.index)),
        (f"Standalone: near pivot support (any touch, <={SUPPORT_PROXIMITY_PCT}%)", near_support),
        (f"Standalone: near MULTI-TOUCH pivot support (>=2x, <={SUPPORT_PROXIMITY_PCT}%)", near_support_multi),
        (f"Standalone: room below pivot resistance (>={RESISTANCE_ROOM_PCT}%)", room_to_resistance),
        ("Standalone: near support AND room to resistance", near_support & room_to_resistance),
        ("SHIPPED: bottoming+momentum+CMF<0+RVOL>=0.8 (reference)", shipped),
        ("Shipped + near pivot support (any touch)", shipped & near_support),
        ("Shipped + near MULTI-TOUCH pivot support", shipped & near_support_multi),
        ("Shipped + room to resistance", shipped & room_to_resistance),
        ("Shipped + near support + room to resistance", shipped & near_support & room_to_resistance),
        ("Shipped + near MULTI-TOUCH support + room to resistance", shipped & near_support_multi & room_to_resistance),
    ]
    results = [evaluate(df, mask, label) for label, mask in candidates]
    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    logger.info("\n%s", results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run()
