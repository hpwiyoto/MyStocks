"""Before building a model: how would the Turnaround v2 candidate rule
(regime in {overextended, bullish} + price_vs_vwap20_pct>5, target +20%/
-10%/60 trading days -- scripts/search_turnaround_v2_target.py +
scripts/tune_turnaround_v2_magnitude.py) have compared against the two
existing model-driven pages, Swing and Turnaround, picking their own
top-2 each day?

Same top-2-per-day methodology as scripts/backtest_top2_screeners.py
(replay each page's own real ranking at every historical as-of date, keep
only the top 2), extended to a 60-trading-day horizon instead of 10 --
the common yardstick here is Turnaround v2's OWN target (+20%/-10%/60
days), not the 10-day one, because the question is specifically "which
of these three would a user have been better off following for a 3-month,
20%-type move" -- Turnaround v2's own target, applied fairly to all three
picks.

This is explicitly NOT re-litigating Swing's established 10-day edge
(69.3% top-2, scripts/backtest_top2_screeners.py) or Turnaround's native
6-month regime-hold precision (~92% POTENSIAL) -- those remain what they
are for THEIR OWN targets. This answers a different, narrower question:
for a 3-month/20%-move goal specifically, which pick would have worked
best.

Turnaround v2's rank key (price_vs_vwap20_pct descending) is a plain
placeholder, not a trained model -- picking a proper rank key (or
training an actual model) is exactly the next step this comparison is
meant to inform, not replace.

Usage:
    python -m scripts.compare_turnaround_v2_top2
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from engine.decision import BUY_THRESHOLD, IHSG_DECLINE_BUY_THRESHOLD
from engine.model import load_model_and_metadata
from engine.predict import MODEL_VERSION as SWING_MODEL_VERSION
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION
from features.regime import BAD_REGIMES
from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import (
    TURNAROUND_THRESHOLD,
    build_feature_matrix,
    load_full_panel,
    load_ihsg_declining_by_date,
    swing_decision_tier,
)
from scripts.search_momentum_rules import WARMUP_DATES, wilson_lower_bound

logger = get_logger("scripts.compare_turnaround_v2_top2")

HORIZON = 60          # trading days -- Turnaround v2's own "maksimum 3 bulan"
TARGET_PCT = 0.20
STOP_PCT = 0.10
AS_OF_STRIDE = 10
V2_REGIMES = {"overextended", "bullish"}


def triple_barrier_outcome(fwd_low: np.ndarray, fwd_high: np.ndarray, entry_price: float) -> int | None:
    """Plain numpy-array scan, not .iterrows() -- with HORIZON=60 (6x
    search_momentum_rules.py's 10-day version) and this called once per
    (ticker, as-of-day) candidate, .iterrows()'s per-call Series
    construction overhead adds up to millions of calls across the full
    backtest; a bare float loop over numpy arrays is meaningfully
    faster."""
    target = entry_price * (1 + TARGET_PCT)
    stop = entry_price * (1 - STOP_PCT)
    for lo, hi in zip(fwd_low, fwd_high):
        if lo <= stop:
            return 0
        if hi >= target:
            return 1
    return None


def run():
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    ticker_frames = {code: g.sort_values("date").reset_index(drop=True) for code, g in df.groupby("stock_code")}
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}
    # Precomputed once per ticker (not re-sliced from the DataFrame on
    # every as-of date) -- the dominant cost here is HORIZON=60 outcome
    # scans across ~900 tickers x ~90 as-of dates, so this matters.
    low_arr = {code: g["low"].to_numpy(dtype=float) for code, g in ticker_frames.items()}
    high_arr = {code: g["high"].to_numpy(dtype=float) for code, g in ticker_frames.items()}

    all_dates = sorted(df["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates, horizon=%d hari, target=+%.0f%%/-%.0f%%", len(as_of_dates), HORIZON, TARGET_PCT * 100, STOP_PCT * 100)

    ihsg_declining_by_date = load_ihsg_declining_by_date(as_of_dates)

    swing_booster, swing_meta = load_model_and_metadata(SWING_MODEL_VERSION)
    turnaround_booster, turnaround_meta = load_model_and_metadata(TURNAROUND_MODEL_VERSION)
    swing_base_rate = swing_meta["base_rate"]
    swing_feature_cols = swing_meta["feature_cols"]
    turnaround_feature_cols = turnaround_meta["feature_cols"]

    picks = []
    for n_done, as_of in enumerate(as_of_dates):
        day_rows = []
        for code, g in ticker_frames.items():
            idx = as_of_idx_by_ticker[code].get(as_of)
            if idx is None or idx < 30:
                continue
            n_fwd = len(g) - (idx + 1)
            if n_fwd < HORIZON:
                continue
            latest = g.iloc[idx]
            if pd.isna(latest.get("rsi_14")) or pd.isna(latest.get("macd_hist")):
                continue
            fwd_low = low_arr[code][idx + 1: idx + 1 + HORIZON]
            fwd_high = high_arr[code][idx + 1: idx + 1 + HORIZON]
            outcome = triple_barrier_outcome(fwd_low, fwd_high, float(latest["close"]))
            if outcome is None:
                continue
            row = latest.to_dict()
            row["stock_code"] = code
            row["outcome"] = outcome
            day_rows.append(row)

        if len(day_rows) < 2:
            continue
        day_df = pd.DataFrame(day_rows)

        buy_threshold = IHSG_DECLINE_BUY_THRESHOLD if ihsg_declining_by_date.get(as_of) else BUY_THRESHOLD
        X_swing = build_feature_matrix(day_df, swing_feature_cols)
        day_df["swing_prob"] = swing_booster.predict(xgb.DMatrix(X_swing))
        day_df["swing_tier"] = [
            swing_decision_tier(p, swing_base_rate, c, buy_threshold)
            for p, c in zip(day_df["swing_prob"], day_df["close"])
        ]

        ta_mask = day_df["regime"].isin(BAD_REGIMES)
        day_df["turnaround_prob"] = np.nan
        if ta_mask.any():
            X_turn = build_feature_matrix(day_df.loc[ta_mask], turnaround_feature_cols)
            day_df.loc[ta_mask, "turnaround_prob"] = turnaround_booster.predict(xgb.DMatrix(X_turn))
        day_df["turnaround_tier"] = np.where(day_df["turnaround_prob"] >= TURNAROUND_THRESHOLD, 0, 1)

        v2_mask = day_df["regime"].isin(V2_REGIMES) & (day_df["price_vs_vwap20_pct"] > 5)

        rankings = {
            "Swing": day_df.sort_values(["swing_tier", "swing_prob"], ascending=[True, False]),
            "Turnaround": day_df.loc[ta_mask].sort_values(["turnaround_tier", "turnaround_prob"], ascending=[True, False]),
            "Turnaround v2 (usulan)": day_df.loc[v2_mask].sort_values("price_vs_vwap20_pct", ascending=False),
        }
        for screener, ranked in rankings.items():
            for rank, (_, r) in enumerate(ranked.head(2).iterrows(), start=1):
                picks.append({"screener": screener, "as_of": as_of, "rank": rank, "stock_code": r["stock_code"], "outcome": r["outcome"]})

        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d picks so far)", n_done + 1, len(as_of_dates), len(picks))

    picks_df = pd.DataFrame(picks)
    logger.info("Done. %d total top-2 picks recorded.", len(picks_df))
    logger.info("=" * 90)
    logger.info("RESULT: top-2-per-day pooled, SEMUA dinilai dengan target Turnaround v2 (+%.0f%%/-%.0f%%/%d hari)",
                TARGET_PCT * 100, STOP_PCT * 100, HORIZON)
    logger.info("=" * 90)
    rows = []
    for screener, g in picks_df.groupby("screener"):
        n_days = g["as_of"].nunique()
        n = len(g)
        wins = int(g["outcome"].sum())
        wr = wins / n if n else float("nan")
        lb = wilson_lower_bound(wins, n) if n else 0.0
        rows.append({"screener": screener, "n_days_with_picks": n_days, "n_picks": n, "win_rate": wr, "wilson_lb": lb})
        for rank, rg in g.groupby("rank"):
            rn, rw = len(rg), int(rg["outcome"].sum())
            logger.info("  [%s] rank=%d n=%d win_rate=%.4f", screener, rank, rn, rw / rn if rn else float("nan"))
    summary = pd.DataFrame(rows).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
