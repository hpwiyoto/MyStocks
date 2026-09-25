"""Direct user follow-up: does layering any of features.expert_rules'
Golden Cross Awal ingredients ON TOP of the already-validated_signal
combo (features.momentum_screener.is_validated_signal) improve it
further -- same question, same method, as how AVWAP was found to help
that combo originally.

IMPORTANT: an earlier run of this exact question (2026-09-25) using the
OLD array-position as-of-date stride found "validated_signal ALONE"
reproduced at n=441/WR=34.24%/LB=29.96% -- NOT the n=523/WR=42.45%/
LB=38.28% documented in scripts/test_strategy_6_criteria.py and shipped
in the app. Tracing it down found the OLD stride
(all_dates[WARMUP_DATES:-HORIZON-1][::AS_OF_STRIDE]) is NOT robust to
insertions/deletions elsewhere in the date array -- deleting 9 phantom
calendar dates (this session's own holiday-phantom-row cleanup) shifted
which ~8% of as-of dates near the tail got selected, with no actual
change in market reality for the rest, and that alone was enough to
swing this narrow/rare filter's count by 15% even though the overall
null-baseline dataset barely moved (-0.07%). Fixed: this script (and
scripts/search_momentum_rules.py, scripts/test_strategy_6_criteria.py)
now use scripts.search_momentum_rules.select_as_of_dates -- calendar-
bucket-anchored, so a deletion elsewhere can only ever perturb the one
bucket it falls in, never cascade. See that function's docstring for
the full story. This script's first candidate reproduces validated_
signal ALONE as a sanity check/current baseline before testing anything
layered on top of it -- treat ITS number as the reference now, not the
one in this docstring's history above.

Calls features.momentum_screener.is_validated_signal and
features.expert_rules.evaluate_expert_signal DIRECTLY (not
re-implementations) so this tests what actually ships on both sides.

Usage:
    python -m scripts.backtest_validated_plus_expert
"""
import datetime as dt

import pandas as pd

from features.expert_rules import evaluate_expert_signal
from features.momentum_screener import compute_avwap_from_low, is_validated_signal
from pipeline.db import get_engine
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import (
    HORIZON,
    LOOKBACK_DAYS,
    select_as_of_dates,
    triple_barrier_outcome,
    wilson_lower_bound,
)

logger = get_logger("scripts.backtest_validated_plus_expert")


def load_full_panel() -> pd.DataFrame:
    engine = get_engine()
    logger.info("Loading full price_history + feature_daily history...")
    df = pd.read_sql(
        """
        SELECT ph.stock_code, ph.date, ph.close, ph.high, ph.low, ph.volume,
               fd.rsi_14, fd.macd, fd.macd_signal, fd.macd_hist, fd.macd_hist_slope_3d,
               fd.cmf_20, fd.rvol_20, fd.regime,
               fd.sma_20, fd.sma_50, fd.sma_200,
               fd.distance_to_support_pct, fd.net_foreign_flow
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
    panel = load_full_panel()
    ticker_frames = {code: g.reset_index(drop=True) for code, g in panel.groupby("stock_code")}
    idx_by_date = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates_dt = sorted(dt.date.fromisoformat(d) for d in panel["date"].unique())
    as_of_dates = [d.isoformat() for d in select_as_of_dates(all_dates_dt)]
    logger.info("%d as-of dates", len(as_of_dates))

    rows = []
    for n_done, as_of in enumerate(as_of_dates):
        for code, g in ticker_frames.items():
            idx = idx_by_date[code].get(as_of)
            if idx is None or idx < 30:
                continue
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window = g.iloc[start:idx + 1]
            latest = window.iloc[-1]
            if pd.isna(latest["rsi_14"]) or pd.isna(latest["macd_hist"]):
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue

            avwap = compute_avwap_from_low(
                window["close"].to_numpy(dtype=float), window["high"].to_numpy(dtype=float),
                window["low"].to_numpy(dtype=float), window["volume"].to_numpy(dtype=float),
            )
            close_above_avwap = bool(latest["close"] >= avwap) if avwap is not None and pd.notna(latest["close"]) else None
            validated = is_validated_signal(
                latest["regime"], latest["macd_hist_slope_3d"], latest["cmf_20"], latest["rvol_20"], close_above_avwap,
            )
            expert = evaluate_expert_signal(window)

            rows.append({
                "as_of": as_of, "stock_code": code, "outcome": outcome,
                "validated_signal": validated,
                "golden_cross": expert["golden_cross"],
                "macd_ok": expert["macd_ok"],
                "rsi_ok": expert["rsi_ok"],
                "liquidity_ok": expert["liquidity_ok"],
                "near_support_bonus_raw": pd.notna(expert["distance_to_support_pct"]) and expert["distance_to_support_pct"] <= 3.0,
                "foreign_flow_positive": expert["foreign_flow_bonus"],
            })
        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rows so far)", n_done + 1, len(as_of_dates), len(rows))

    out = pd.DataFrame(rows)
    logger.info("Dataset ready: %d resolved (ticker, as-of-date) rows", len(out))
    return out


def evaluate(df: pd.DataFrame, mask: pd.Series, label: str, null_rate: float) -> dict:
    sub = df[mask]
    n = len(sub)
    wins = int(sub["outcome"].sum())
    wr = wins / n if n else float("nan")
    return {
        "label": label, "n": n, "win_rate": wr,
        "lift_vs_null": wr / null_rate if n and null_rate else float("nan"),
        "wilson_lb": wilson_lower_bound(wins, n) if n else 0.0,
    }


def run():
    df = build_dataset()
    if df.empty:
        logger.error("Dataset is empty -- nothing to evaluate.")
        return
    null_rate = df["outcome"].mean()
    logger.info("=" * 70)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(df), null_rate)
    logger.info("=" * 70)

    v = df["validated_signal"]
    candidates = [
        ("validated_signal ALONE (reproduces the 42.45%/523/38.28% baseline)", v),
        ("validated_signal + golden_cross (any type)", v & df["golden_cross"]),
        ("validated_signal + macd_ok (expert early-stage)", v & df["macd_ok"]),
        ("validated_signal + rsi_ok (40-70)", v & df["rsi_ok"]),
        ("validated_signal + liquidity_ok (>=Rp1M/hari)", v & df["liquidity_ok"]),
        ("validated_signal + near support (<=3%)", v & df["near_support_bonus_raw"]),
        ("validated_signal + foreign_flow positive", v & df["foreign_flow_positive"]),
        ("validated_signal + golden_cross + near_support", v & df["golden_cross"] & df["near_support_bonus_raw"]),
    ]

    results = [evaluate(df, mask, label, null_rate) for label, mask in candidates]
    results_df = pd.DataFrame(results).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 65)
    logger.info("\n%s", results_df.to_string(index=False))

    logger.info("=" * 70)
    logger.info("Ranked by Wilson 95%% lower bound (conservative -- penalizes small n, "
                "not just the raw point estimate)")
    logger.info("=" * 70)


if __name__ == "__main__":
    run()
