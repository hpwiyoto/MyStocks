"""User-proposed screener rule, backtested and compared head-to-head with
Swing / Turnaround / Momentum.

The rule (an "early reversal, buy/watch before MACD confirms" setup):
  1. MACD histogram is RED  -> macd_hist < 0. (This is identical to "MACD
     line still below its signal line", so the user's "act BEFORE the MACD
     line crosses signal from below" is already baked in -- no separate
     condition.)
  2. EMA9 has been BELOW MA20 for at least 9 consecutive trading days
     -> (ema_9 < sma_20) true on each of the last 9 bars. Downtrend
     structure still intact.
  3. Stochastic RSI %K JUST crossed above %D from below -> k[t] > d[t] and
     k[t-1] <= d[t-1]. A leading oscillator turn, ahead of the lagging
     MACD-line/signal cross.

Common yardstick + methodology are the SAME as every other screener
backtest here (search_momentum_rules.py's 10-day/+5%/-2.5% triple-barrier
outcome, 5-year no-lookahead as-of-date sampling), and the Swing /
Turnaround / Momentum top-2 numbers are recomputed in THIS run on the
identical as-of dates (reusing scripts/backtest_top2_screeners.py's exact
scoring/ranking helpers) so the comparison is genuinely same-footing, not
against numbers from a prior run.

StochRSI isn't stored in feature_daily -- computed here from close via
ta.momentum.StochRSIIndicator (window=14, smooth1=3, smooth2=3, the
standard convention). Same optimism caveat as backtest_triple_
intersection.py for any Swing/Turnaround probability involved.

Usage:
    python -m scripts.test_stochrsi_early_reversal_rule
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from ta.momentum import StochRSIIndicator

from engine.decision import BUY_THRESHOLD, IHSG_DECLINE_BUY_THRESHOLD
from engine.model import load_model_and_metadata
from engine.predict import MODEL_VERSION as SWING_MODEL_VERSION
from engine.predict_turnaround import MODEL_VERSION as TURNAROUND_MODEL_VERSION
from features.momentum_screener import DEFAULT_REGIME_PRIORITY, REGIME_PRIORITY, classify_macd_status, detect_bullish_divergence
from features.regime import BAD_REGIMES
from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import (
    TURNAROUND_THRESHOLD,
    build_feature_matrix,
    load_full_panel,
    load_ihsg_declining_by_date,
    momentum_extra_indicators,
    swing_decision_tier,
)
from scripts.search_momentum_rules import AS_OF_STRIDE, HORIZON, LOOKBACK_DAYS, WARMUP_DATES, triple_barrier_outcome, wilson_lower_bound

logger = get_logger("scripts.test_stochrsi_early_reversal_rule")

EMA_BELOW_MA_MIN_DAYS = 9


def add_rule_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """g: one ticker, ascending by date, already has ema_9/sma_20/macd_hist
    from feature_daily + close from price_history. Adds the three
    sub-conditions and the full-combo flag, all no-lookahead (rolling/lag
    only)."""
    g = g.copy()
    sr = StochRSIIndicator(close=g["close"], window=14, smooth1=3, smooth2=3)
    k, d = sr.stochrsi_k(), sr.stochrsi_d()
    g["stochrsi_cross_up"] = (k > d) & (k.shift(1) <= d.shift(1))

    ema_below = (g["ema_9"] < g["sma_20"]).astype("float")
    # all of the last EMA_BELOW_MA_MIN_DAYS bars below -> rolling sum == N
    g["ema9_below_ma20_streak_ok"] = ema_below.rolling(EMA_BELOW_MA_MIN_DAYS).sum() >= EMA_BELOW_MA_MIN_DAYS

    g["macd_hist_red"] = g["macd_hist"] < 0
    g["rule_hit"] = g["macd_hist_red"] & g["ema9_below_ma20_streak_ok"] & g["stochrsi_cross_up"]
    return g


def run():
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    ticker_frames = {
        code: add_rule_indicators(momentum_extra_indicators(g.sort_values("date").reset_index(drop=True)))
        for code, g in df.groupby("stock_code")
    }
    as_of_idx_by_ticker = {code: {dd: i for i, dd in enumerate(g["date"])} for code, g in ticker_frames.items()}

    all_dates = sorted(df["date"].unique())
    usable_dates = all_dates[WARMUP_DATES:-HORIZON - 1]
    as_of_dates = usable_dates[::AS_OF_STRIDE]
    logger.info("%d as-of dates", len(as_of_dates))
    ihsg_declining_by_date = load_ihsg_declining_by_date(as_of_dates)

    swing_booster, swing_meta = load_model_and_metadata(SWING_MODEL_VERSION)
    turnaround_booster, turnaround_meta = load_model_and_metadata(TURNAROUND_MODEL_VERSION)
    swing_base_rate = swing_meta["base_rate"]
    swing_feature_cols = swing_meta["feature_cols"]
    turnaround_feature_cols = turnaround_meta["feature_cols"]

    rule_rows = []           # every rule_hit instance -> for the standalone edge check
    subcond_rows = []        # for the piece-by-piece breakdown
    top2 = {s: [] for s in ["Rule (proposed)", "Swing", "Turnaround", "Momentum"]}

    for n_done, as_of in enumerate(as_of_dates):
        day_rows = []
        for code, g in ticker_frames.items():
            idx = as_of_idx_by_ticker[code].get(as_of)
            if idx is None or idx < 30:
                continue
            fwd = g.iloc[idx + 1: idx + 1 + HORIZON]
            if len(fwd) < HORIZON:
                continue
            latest = g.iloc[idx]
            if pd.isna(latest.get("rsi_14")) or pd.isna(latest.get("macd_hist")):
                continue
            outcome = triple_barrier_outcome(fwd, float(latest["close"]))
            if outcome is None:
                continue
            row = latest.to_dict()
            row["stock_code"] = code
            row["outcome"] = outcome
            start = max(0, idx - LOOKBACK_DAYS + 1)
            window_df = g.iloc[start:idx + 1]
            row.update(detect_bullish_divergence(window_df))
            row["macd_status"] = classify_macd_status(window_df["macd_hist"])
            day_rows.append(row)
        if len(day_rows) < 2:
            continue
        day_df = pd.DataFrame(day_rows)

        buy_threshold = IHSG_DECLINE_BUY_THRESHOLD if ihsg_declining_by_date.get(as_of) else BUY_THRESHOLD
        day_df["swing_prob"] = swing_booster.predict(xgb.DMatrix(build_feature_matrix(day_df, swing_feature_cols)))
        day_df["swing_tier"] = [swing_decision_tier(p, swing_base_rate, c, buy_threshold) for p, c in zip(day_df["swing_prob"], day_df["close"])]

        ta_mask = day_df["regime"].isin(BAD_REGIMES)
        day_df["turnaround_prob"] = np.nan
        if ta_mask.any():
            day_df.loc[ta_mask, "turnaround_prob"] = turnaround_booster.predict(xgb.DMatrix(build_feature_matrix(day_df.loc[ta_mask], turnaround_feature_cols)))
        day_df["turnaround_tier"] = np.where(day_df["turnaround_prob"] >= TURNAROUND_THRESHOLD, 0, 1)

        avwap_ok = day_df["close_above_avwap"] == True  # noqa: E712
        day_df["validated_signal"] = (
            (day_df["regime"] == "bottoming") & (day_df["macd_hist_slope_3d"] > 0)
            & (day_df["cmf_20"] < 0) & (day_df["rvol_20"] >= 0.8) & avwap_ok
        )
        day_df["divergence_tier"] = np.select(
            [day_df["divergence_rsi"] & day_df["divergence_macd"], day_df["divergence_rsi"] | day_df["divergence_macd"]],
            [0, 1], default=2,
        )
        day_df["regime_priority"] = day_df["regime"].map(REGIME_PRIORITY).fillna(DEFAULT_REGIME_PRIORITY).astype(int)
        day_df["rsi_pivot_distance"] = (day_df["rsi_14"] - 50).abs()

        # --- proposed rule: record every hit (standalone) + sub-conditions ---
        for _, r in day_df.iterrows():
            subcond_rows.append({
                "macd_hist_red": bool(r["macd_hist_red"]),
                "ema9_below_ma20_streak_ok": bool(r["ema9_below_ma20_streak_ok"]),
                "stochrsi_cross_up": bool(r["stochrsi_cross_up"]),
                "rule_hit": bool(r["rule_hit"]),
                "outcome": r["outcome"],
            })
        rule_pool = day_df[day_df["rule_hit"].fillna(False)]
        for _, r in rule_pool.iterrows():
            rule_rows.append({"as_of": as_of, "stock_code": r["stock_code"], "outcome": r["outcome"]})

        # --- top-2 per day, all four on identical footing ---
        rankings = {
            "Rule (proposed)": rule_pool.sort_values("swing_prob", ascending=False),
            "Swing": day_df.sort_values(["swing_tier", "swing_prob"], ascending=[True, False]),
            "Turnaround": day_df.loc[ta_mask].sort_values(["turnaround_tier", "turnaround_prob"], ascending=[True, False]),
            "Momentum": day_df.sort_values(
                ["validated_signal", "divergence_tier", "regime_priority", "rsi_pivot_distance", "divergence_age_days", "swing_prob"],
                ascending=[False, True, True, True, True, False], na_position="last",
            ),
        }
        for screener, ranked in rankings.items():
            for _, r in ranked.head(2).iterrows():
                top2[screener].append(r["outcome"])

        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done (%d rule hits so far)", n_done + 1, len(as_of_dates), len(rule_rows))

    subcond = pd.DataFrame(subcond_rows)
    null_rate = subcond["outcome"].mean()
    logger.info("=" * 74)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(subcond), null_rate)
    logger.info("=" * 74)

    def line(label, mask):
        sub = subcond[mask]
        n = len(sub)
        w = int(sub["outcome"].sum())
        wr = w / n if n else float("nan")
        lb = wilson_lower_bound(w, n) if n else 0.0
        logger.info("  %-52s n=%-6d win_rate=%.4f  wilson_lb=%.4f", label, n, wr, lb)

    logger.info("PROPOSED RULE -- each condition alone, then the full combo:")
    line("1. MACD histogram merah (macd_hist < 0)", subcond["macd_hist_red"])
    line(f"2. EMA9 < MA20 selama >= {EMA_BELOW_MA_MIN_DAYS} hari", subcond["ema9_below_ma20_streak_ok"])
    line("3. StochRSI %K cross %D dari bawah (fresh)", subcond["stochrsi_cross_up"])
    line("FULL RULE (1 AND 2 AND 3)", subcond["rule_hit"])

    logger.info("=" * 74)
    logger.info("TOP-2 PER DAY -- proposed rule vs the three screeners, identical as-of dates")
    logger.info("=" * 74)
    rows = []
    for screener, outs in top2.items():
        n = len(outs)
        w = int(sum(outs))
        wr = w / n if n else float("nan")
        lb = wilson_lower_bound(w, n) if n else 0.0
        rows.append({"screener": screener, "n_picks": n, "win_rate": round(wr, 4), "wilson_lb": round(lb, 4)})
    summary = pd.DataFrame(rows).sort_values("wilson_lb", ascending=False)
    pd.set_option("display.width", 200)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
