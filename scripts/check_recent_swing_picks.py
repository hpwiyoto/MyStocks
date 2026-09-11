"""Real-money-style spot check, not a statistical backtest: take the exact
Swing top-3 pick from N trading days ago (default 10 -- the horizon Swing
itself targets) and look at what ACTUALLY happened to those specific 3
tickers since, using real price_history bars that have already occurred.

Different in kind from scripts/backtest_top2_screeners.py's top-2 pooled
win rate (109 as-of dates, hundreds of picks) -- this is n=3, one date,
answering a concrete "did THIS specific recommendation pay off" question
rather than "how does this selection policy do on average". Not a
substitute for the pooled backtest's statistical confidence; a genuine
answer either way here is still just one data point.

Same scoring pipeline as backtest_top2_screeners.py (current, fully-
trained Swing model against that day's real feature row -- same optimism
caveat: this replays today's model against the past, not a true
point-in-time snapshot of what the model looked like N days ago) and the
same +5%/-2.5%/10-trading-day triple-barrier convention used everywhere
else in this project, PLUS the raw achieved max/min % move (a triple-
barrier outcome of "still open, hit neither barrier" is not very
satisfying for a plain "did it work" question -- the raw move answers
that even when the formal barrier hasn't resolved yet).

Usage:
    python -m scripts.check_recent_swing_picks [n_days_ago] [top_n]
    (defaults: 10 trading days ago, top 3)
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from engine.decision import BUY_THRESHOLD, IHSG_DECLINE_BUY_THRESHOLD
from engine.model import load_model_and_metadata
from engine.predict import MODEL_VERSION as SWING_MODEL_VERSION
from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import (
    build_feature_matrix,
    load_full_panel,
    load_ihsg_declining_by_date,
    swing_decision_tier,
)
from scripts.search_momentum_rules import STOP_PCT, TARGET_PCT, triple_barrier_outcome

logger = get_logger("scripts.check_recent_swing_picks")

TIER_LABEL = {0: "BUY", 1: "WATCH", 2: "AVOID"}


def run(n_days_ago: int = 10, top_n: int = 3):
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    ticker_frames = {code: g.sort_values("date").reset_index(drop=True) for code, g in df.groupby("stock_code")}

    all_dates = sorted(df["date"].unique())
    if len(all_dates) <= n_days_ago:
        logger.error("Not enough trading dates in price_history to look %d days back.", n_days_ago)
        return
    as_of = all_dates[-1 - n_days_ago]  # exactly n_days_ago trading days before the latest date we have
    logger.info("As-of date: %s (today's data goes up to %s, so this has %d realized trading days after it)",
                as_of.date(), all_dates[-1].date(), n_days_ago)

    ihsg_declining = load_ihsg_declining_by_date([as_of])[as_of]
    buy_threshold = IHSG_DECLINE_BUY_THRESHOLD if ihsg_declining else BUY_THRESHOLD
    logger.info("IHSG declining on that date: %s -> BUY threshold %.2f", ihsg_declining, buy_threshold)

    swing_booster, swing_meta = load_model_and_metadata(SWING_MODEL_VERSION)
    swing_base_rate = swing_meta["base_rate"]
    swing_feature_cols = swing_meta["feature_cols"]

    day_rows = []
    for code, g in ticker_frames.items():
        matches = g.index[g["date"] == as_of]
        if len(matches) == 0:
            continue
        idx = matches[0]
        if idx < 30 or pd.isna(g.loc[idx, "rsi_14"]) or pd.isna(g.loc[idx, "macd_hist"]):
            continue
        row = g.loc[idx].to_dict()
        row["stock_code"] = code
        row["_idx"] = idx
        day_rows.append(row)

    if not day_rows:
        logger.error("No scoreable tickers on %s.", as_of.date())
        return
    day_df = pd.DataFrame(day_rows)

    X = build_feature_matrix(day_df, swing_feature_cols)
    day_df["swing_prob"] = swing_booster.predict(xgb.DMatrix(X))
    day_df["swing_tier"] = [
        swing_decision_tier(p, swing_base_rate, c, buy_threshold)
        for p, c in zip(day_df["swing_prob"], day_df["close"])
    ]
    ranked = day_df.sort_values(["swing_tier", "swing_prob"], ascending=[True, False]).reset_index(drop=True)
    top = ranked.head(top_n)

    logger.info("=" * 100)
    logger.info("SWING TOP %d on %s (as the real page would have ranked them that day):", top_n, as_of.date())
    logger.info("=" * 100)

    wins, resolved = 0, 0
    for _, r in top.iterrows():
        code = r["stock_code"]
        g = ticker_frames[code]
        idx = int(r["_idx"])
        entry_price = float(r["close"])
        fwd = g.iloc[idx + 1: idx + 1 + n_days_ago]
        n_fwd = len(fwd)
        if n_fwd == 0:
            logger.info("  %-5s  tier=%-5s prob=%.3f  entry=%.0f  -- belum ada data setelahnya sama sekali",
                        code, TIER_LABEL[int(r["swing_tier"])], r["swing_prob"], entry_price)
            continue
        max_high = float(fwd["high"].max())
        min_low = float(fwd["low"].min())
        last_close = float(fwd["close"].iloc[-1])
        max_gain_pct = (max_high / entry_price - 1) * 100
        max_drawdown_pct = (min_low / entry_price - 1) * 100
        actual_close_ret_pct = (last_close / entry_price - 1) * 100
        outcome = triple_barrier_outcome(fwd, entry_price) if n_fwd >= n_days_ago else None
        if outcome is not None:
            resolved += 1
            wins += outcome
            verdict = f"MENANG (sentuh +{TARGET_PCT*100:.1f}% duluan)" if outcome == 1 else f"KALAH (sentuh stop -{STOP_PCT*100:.1f}% duluan)"
        else:
            hit_target = max_gain_pct >= TARGET_PCT * 100
            verdict = ("belum tuntas (data forward < %d hari)" % n_days_ago if n_fwd < n_days_ago
                        else "belum tuntas (10 hari lewat, tidak sentuh target ATAU stop)")
            if hit_target:
                verdict += " -- tapi harga TERTINGGI sudah melewati +5%"
        logger.info(
            "  %-5s  tier=%-5s prob=%.3f  entry=%.0f (%s)  -> %d hari data: tertinggi %+.1f%%, terendah %+.1f%%, "
            "close hari ke-%d %+.1f%%  |  %s",
            code, TIER_LABEL[int(r["swing_tier"])], r["swing_prob"], entry_price, as_of.date(),
            n_fwd, max_gain_pct, max_drawdown_pct, n_fwd, actual_close_ret_pct, verdict,
        )

    if resolved:
        logger.info("-" * 100)
        logger.info("Ringkasan (hanya yang formal tuntas, sentuh target ATAU stop): %d/%d menang", wins, resolved)
    logger.info("Catatan: model Swing yang dipakai untuk skor tanggal %s adalah versi TERLATIH SAAT INI, "
                "bukan snapshot model persis %d hari lalu -- sama seperti caveat di backtest_top2_screeners.py.",
                as_of.date(), n_days_ago)


if __name__ == "__main__":
    n_days_ago_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    top_n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(n_days_ago_arg, top_n_arg)
