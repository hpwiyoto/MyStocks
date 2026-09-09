"""Does a feature that contributes heavily to the Swing/Turnaround MODELS
(XGBoost gain >3%%) also work as a simple threshold FILTER in the Momentum
Screener?

This is a genuinely open question, not an assumed "yes" -- a feature's
gain contribution measures its usefulness across hundreds of trees, each
splitting on it at a DIFFERENT threshold in combination with dozens of
other features non-linearly. That is a fundamentally different claim from
"there exists ONE simple threshold that, used ALONE (or ANDed onto the
shipped rule), beats the null baseline as a hand-readable filter." Already
falsified once in this exact codebase: distance_to_resistance_pct/
distance_to_support_pct contribute 4.24%%/3.24%% to the Swing model, yet a
much MORE sophisticated version of that same idea (pivot-based multi-touch
support/resistance, features/support_resistance.py) was tested as a
Momentum Screener filter and REJECTED (scripts/test_pivot_support_
resistance.py) -- every variant scored a worse Wilson LB than the shipped
rule alone.

Candidates here are every OTHER Swing/Turnaround feature currently NOT
used anywhere in Momentum Screener that contributes >3%% gain to either
model (checked directly against models/direction_xgboost_v5.json and
models/turnaround_xgboost_v1.json on 2026-09-09 -- see the commit message
for the exact ranking). Excludes rsi_14/rsi_distance_50 (already used as
a RANKING tiebreaker, not tested again as a filter) and pure valuation
features (trailing_pe, price_to_book, market_cap_log -- >3%% in Turnaround
but a fundamentally different signal category than a MOMENTUM screener,
out of scope here) and fundamentally-a-calendar-effect is_month_end_week
(technically >3%% in Swing but not a "momentum" criterion by any
reasonable definition, mentioned in the writeup instead of tested).

  mfi_14                (Turnaround's single largest contributor: 21.1%%)
  overnight_gap_pct     (Swing: 5.5%%)
  ret_10d_atr_norm      (Swing: 5.1%%, newest feature, landed 2026-09-08)
  ret_10d_pct           (Swing: 3.6%%)
  price_vs_sma50_pct    (Turnaround: 4.5%%)
  price_vs_vwap20_pct   (Swing: 3.2%% -- NOT the same indicator as the
                          already-adopted Anchored-VWAP-from-50d-low;
                          this is a plain rolling 20-day VWAP)

Direction isn't obvious for most of these up front, so both directions
are tested for each (ambiguous ones) or the more natural single direction
(clearly-directional ones e.g. MFI oversold) -- same no-lookahead 5-year
walk-forward dataset/methodology as every prior Momentum Screener
backtest (scripts/search_momentum_rules.py's harness).

Usage:
    python -m scripts.test_top_model_features_as_criteria
"""
import numpy as np
import pandas as pd

from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import (
    AS_OF_STRIDE,
    HORIZON,
    LOOKBACK_DAYS,
    WARMUP_DATES,
    triple_barrier_outcome,
    wilson_lower_bound,
)

logger = get_logger("scripts.test_top_model_features_as_criteria")

EXTRA_COLS = ["mfi_14", "overnight_gap_pct", "ret_10d_atr_norm", "ret_10d_pct", "price_vs_sma50_pct", "price_vs_vwap20_pct"]


def load_panel_with_extra_features() -> pd.DataFrame:
    engine = get_engine()
    logger.info("Loading price_history + feature_daily (incl. candidate columns)...")
    df = pd.read_sql(
        f"""
        SELECT ph.stock_code, ph.date, ph.close, ph.high, ph.low, ph.volume,
               fd.rsi_14, fd.macd_hist, fd.macd_hist_slope_3d,
               fd.cmf_20, fd.rvol_20, fd.regime,
               {", ".join(f"fd.{c}" for c in EXTRA_COLS)}
        FROM price_history ph
        JOIN feature_daily fd ON fd.stock_code = ph.stock_code AND fd.date = ph.date
        WHERE ph.source_provider = 'yfinance'
        ORDER BY ph.stock_code, ph.date
        """,
        engine,
    )
    logger.info("Loaded %d rows across %d tickers", len(df), df["stock_code"].nunique())
    return df


def build_dataset() -> pd.DataFrame:
    panel = load_panel_with_extra_features()
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
            latest = g.iloc[start:idx + 1].iloc[-1]
            if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]):
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue
            row = {
                "as_of": as_of, "stock_code": code,
                "regime": latest["regime"], "macd_hist_slope_3d": latest["macd_hist_slope_3d"],
                "cmf_20": latest["cmf_20"], "rvol_20": latest["rvol_20"], "outcome": outcome,
            }
            for c in EXTRA_COLS:
                row[c] = latest[c]
            rows.append(row)
        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rows so far)", n_done + 1, len(as_of_dates), len(rows))

    out = pd.DataFrame(rows)
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

    shipped = (
        (df["regime"] == "bottoming")
        & (df["macd_hist_slope_3d"] > 0)
        & (df["cmf_20"] < 0)
        & (df["rvol_20"] >= 0.8)
    )

    candidates = [("SHIPPED (reference)", shipped)]

    def add_both_directions(col, label):
        has = df[col].notna()
        candidates.append((f"Standalone: {label} < 0", has & (df[col] < 0)))
        candidates.append((f"Standalone: {label} > 0", has & (df[col] > 0)))
        candidates.append((f"Shipped + {label} < 0", shipped & has & (df[col] < 0)))
        candidates.append((f"Shipped + {label} > 0", shipped & has & (df[col] > 0)))

    # MFI: directional hypothesis is clear (oversold = potential bounce,
    # same reasoning as RSI<45 in the 6-criteria test) -- two thresholds,
    # not two directions.
    has_mfi = df["mfi_14"].notna()
    candidates.append(("Standalone: MFI < 30 (oversold)", has_mfi & (df["mfi_14"] < 30)))
    candidates.append(("Standalone: MFI < 50", has_mfi & (df["mfi_14"] < 50)))
    candidates.append(("Shipped + MFI < 30", shipped & has_mfi & (df["mfi_14"] < 30)))
    candidates.append(("Shipped + MFI < 50", shipped & has_mfi & (df["mfi_14"] < 50)))

    add_both_directions("overnight_gap_pct", "overnight_gap_pct")
    add_both_directions("ret_10d_atr_norm", "ret_10d_atr_norm")
    add_both_directions("ret_10d_pct", "ret_10d_pct")
    add_both_directions("price_vs_sma50_pct", "price_vs_sma50_pct")
    add_both_directions("price_vs_vwap20_pct", "price_vs_vwap20_pct")

    results = [evaluate(df, mask, label) for label, mask in candidates]
    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    logger.info("\n%s", results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run()
