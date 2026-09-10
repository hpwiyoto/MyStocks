"""Does restricting the app's universe to LIQUID emitten only (average
daily traded value >= a threshold) improve each screener's top-2 accuracy?

User's reasoning: a thinly-traded stock's daily close can be one lot, its
"+5%% move" may be untradeable, and RSI/MACD/CMF computed on sparse volume
are less meaningful -- so filtering to liquid names should make the
signals reflect a real, executable situation. The gocap floor
(engine.decision.GOCAP_PRICE_FLOOR = Rp50) only guards PRICE, not
liquidity -- a Rp2000 stock can still trade almost nothing.

This reuses scripts/backtest_top2_screeners.py's exact scoring/ranking
machinery (same 5-year no-lookahead dataset, same per-page sort keys,
same 10-day/+5%%/-2.5%% common yardstick) and adds one dimension: before
ranking, the candidate pool is gated on trailing-60-trading-day average
traded value (close * volume, in Rupiah). Every expensive step (panel
load, model scoring) runs ONCE; only the cheap pool-filter -> rank ->
head(2) is repeated per threshold, so all thresholds are measured on
identical underlying scores.

Thresholds tested: none, Rp0.5b, Rp1b, Rp2b, Rp5b per day. From the
universe's own distribution (checked 2026-09-10): median ticker trades
~Rp0.7b/day, so Rp1b keeps ~45%%, Rp5b keeps ~24%%.

Usage:
    python -m scripts.test_liquidity_filter
"""
import numpy as np
import pandas as pd
import xgboost as xgb

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

logger = get_logger("scripts.test_liquidity_filter")

LIQUIDITY_WINDOW = 60          # trailing trading days for the avg-traded-value estimate
THRESHOLDS_RP = [0.0, 0.5e9, 1e9, 2e9, 5e9]  # "0.0" = no liquidity filter (baseline)


def run():
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    df["traded_value"] = df["close"] * df["volume"]

    ticker_frames = {}
    for code, g in df.groupby("stock_code"):
        g = momentum_extra_indicators(g.sort_values("date").reset_index(drop=True))
        # Trailing avg traded value, shifted by 1 so the as-of day itself
        # is NOT in its own liquidity estimate (no lookahead -- decide
        # "was this liquid GOING IN" using only prior days).
        g["avg_traded_value_60d"] = g["traded_value"].rolling(LIQUIDITY_WINDOW, min_periods=20).mean().shift(1)
        ticker_frames[code] = g
    as_of_idx_by_ticker = {code: {d: i for i, d in enumerate(g["date"])} for code, g in ticker_frames.items()}

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

    # picks[threshold][screener] = list of outcomes
    picks = {thr: {s: [] for s in ["Swing", "Turnaround", "Momentum", "Rekomendasi Emitten"]} for thr in THRESHOLDS_RP}

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
        X_swing = build_feature_matrix(day_df, swing_feature_cols)
        day_df["swing_prob"] = swing_booster.predict(xgb.DMatrix(X_swing))
        day_df["swing_tier"] = [swing_decision_tier(p, swing_base_rate, c, buy_threshold) for p, c in zip(day_df["swing_prob"], day_df["close"])]

        ta_mask = day_df["regime"].isin(BAD_REGIMES)
        day_df["turnaround_prob"] = np.nan
        if ta_mask.any():
            X_turn = build_feature_matrix(day_df.loc[ta_mask], turnaround_feature_cols)
            day_df.loc[ta_mask, "turnaround_prob"] = turnaround_booster.predict(xgb.DMatrix(X_turn))
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
        day_df["swing_hit"] = day_df["swing_tier"] <= 1
        day_df["turnaround_hit"] = day_df["turnaround_tier"] == 0
        day_df["momentum_hit"] = day_df["validated_signal"].fillna(False)
        day_df["agreement_count"] = day_df[["swing_hit", "turnaround_hit", "momentum_hit"]].sum(axis=1).astype(int)
        day_df["combined_score"] = day_df["swing_prob"].fillna(0) + day_df["turnaround_prob"].fillna(0)
        day_df["liq"] = day_df["avg_traded_value_60d"]

        for thr in THRESHOLDS_RP:
            # NaN liquidity (a ticker without 20+ prior days) is treated as
            # "not known to be liquid" -> excluded by any real threshold,
            # kept only at thr=0 (the no-filter baseline).
            liq_ok = day_df["liq"].notna() & (day_df["liq"] >= thr) if thr > 0 else pd.Series(True, index=day_df.index)
            pool = day_df[liq_ok]
            if pool.empty:
                continue
            rankings = {
                "Swing": pool.sort_values(["swing_tier", "swing_prob"], ascending=[True, False]),
                "Turnaround": pool[pool["regime"].isin(BAD_REGIMES)].sort_values(["turnaround_tier", "turnaround_prob"], ascending=[True, False]),
                "Momentum": pool.sort_values(
                    ["validated_signal", "divergence_tier", "regime_priority", "rsi_pivot_distance", "divergence_age_days", "swing_prob"],
                    ascending=[False, True, True, True, True, False], na_position="last",
                ),
                "Rekomendasi Emitten": pool[pool["agreement_count"] >= 2].sort_values(["agreement_count", "combined_score"], ascending=[False, False]),
            }
            for screener, ranked in rankings.items():
                for _, r in ranked.head(2).iterrows():
                    picks[thr][screener].append(r["outcome"])

        if (n_done + 1) % 20 == 0:
            logger.info("... %d/%d as-of dates done", n_done + 1, len(as_of_dates))

    logger.info("=" * 78)
    logger.info("RESULT: top-2 pooled win rate (Wilson LB in parens), by screener x liquidity threshold")
    logger.info("=" * 78)
    rows = []
    for screener in ["Swing", "Turnaround", "Momentum", "Rekomendasi Emitten"]:
        row = {"screener": screener}
        for thr in THRESHOLDS_RP:
            outs = picks[thr][screener]
            n = len(outs)
            wins = int(sum(outs))
            wr = wins / n if n else float("nan")
            lb = wilson_lower_bound(wins, n) if n else 0.0
            label = "none" if thr == 0 else f">=Rp{thr/1e9:g}b"
            row[label] = f"{wr*100:.1f}% (LB {lb*100:.1f}, n={n})"
        rows.append(row)
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
