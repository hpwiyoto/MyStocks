"""What should REPLACE the existing Turnaround page's target?

User's ask: horizon down from 6 months to a MAX of 3 months (60 trading
days); the gain doesn't need to be a bagger, moderate is fine; and --
critically -- do NOT presume the starting point has to be bottoming/
bearish/early_reversal. scripts/test_turnaround_early_bounce.py already
showed that presumption doesn't hold for a 3-month magnitude target
(bottoming/sideways alone was slightly WORSE than random). So this script
answers the prior question properly: across ALL 7 regimes, with a real
(not "touch-the-high") triple-barrier outcome, which starting regime(s)
actually have edge for a 3-month target -- then searches technical
conditions within the best regime(s) found.

Target definition: +15% / -7.5% / 60 trading days. A genuine triple
barrier (first-touch-wins, using the forward LOW/HIGH path, not just "did
it ever touch the high") -- this is the fix for early_bounce.py's
disqualifying finding (spikes that reverse look like wins under a
touch-the-high measure but are losses under a real barrier). Same 2:1
reward:risk ratio as Swing's 5%/2.5%, scaled up roughly with sqrt(time)
for a ~6x longer horizon (sqrt(6)=~2.45, 5%*2.45=12-15%) -- a reasoned
starting point, not the final word; once a promising regime/condition
combo is found, the magnitude itself is worth re-tuning.

Stage A: null baseline + per-regime breakdown (which regime(s) actually
beat random for THIS target, unlike presuming bottoming).
Stage B: within the best regime(s), sweep individual technical conditions
(available feature_daily columns) for which ones separate winners.

Usage:
    python -m scripts.search_turnaround_v2_target
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import load_full_panel
from scripts.search_momentum_rules import wilson_lower_bound

logger = get_logger("scripts.search_turnaround_v2_target")

HORIZON = 60          # trading days -- "maksimum 3 bulan"
TARGET_PCT = 0.15
STOP_PCT = 0.075
WARMUP = 60
AS_OF_STRIDE = 10
ALL_REGIMES = ["bullish", "early_reversal", "accumulation", "sideways", "bottoming", "bearish", "overextended"]


def build_dataset() -> pd.DataFrame:
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= WARMUP + HORIZON:
            continue
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        valid_n = n - HORIZON
        fwd_high = sliding_window_view(high[1:], HORIZON)  # shape (valid_n, HORIZON); row i = high[i+1:i+1+HORIZON]
        fwd_low = sliding_window_view(low[1:], HORIZON)
        idxs = np.arange(WARMUP, valid_n)
        if len(idxs) == 0:
            continue
        entry = close[idxs]
        target_hit = fwd_high[idxs] >= (entry[:, None] * (1 + TARGET_PCT))
        stop_hit = fwd_low[idxs] <= (entry[:, None] * (1 - STOP_PCT))
        target_any = target_hit.any(axis=1)
        stop_any = stop_hit.any(axis=1)
        first_target_t = np.where(target_any, target_hit.argmax(axis=1), HORIZON)
        first_stop_t = np.where(stop_any, stop_hit.argmax(axis=1), HORIZON)
        resolved = target_any | stop_any
        win = first_target_t < first_stop_t  # a same-day tie counts as stop (conservative), matches
        outcome = np.where(resolved, win.astype(float), np.nan)

        sub = g.iloc[WARMUP:valid_n].copy()
        sub["outcome"] = outcome
        sub["stock_code"] = code
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["stock_code", "date"])
    out["_rank"] = out.groupby("stock_code").cumcount()
    out = out[out["_rank"] % AS_OF_STRIDE == 0].drop(columns="_rank")
    out = out.dropna(subset=["outcome"])  # drop still-open (unresolved within HORIZON) instances
    logger.info("Dataset: %d resolved (ticker, as-of) rows across %d tickers", len(out), out["stock_code"].nunique())
    return out


def _stat(sub: pd.DataFrame) -> tuple[int, float, float]:
    n = len(sub)
    if n == 0:
        return 0, float("nan"), 0.0
    w = int(sub["outcome"].sum())
    return n, w / n, wilson_lower_bound(w, n)


def run():
    df = build_dataset()
    n0, wr0, lb0 = _stat(df)
    logger.info("=" * 100)
    logger.info("TARGET: +%.1f%% / -%.1f%% / %d hari perdagangan (triple-barrier asli, bukan touch-the-high)",
                TARGET_PCT * 100, STOP_PCT * 100, HORIZON)
    logger.info("NULL BASELINE (semua regime, tanpa syarat apapun): n=%d win_rate=%.1f%% wilson_lb=%.1f%%", n0, wr0 * 100, lb0 * 100)
    logger.info("=" * 100)

    logger.info("STAGE A -- breakdown per REGIME (regime SAAT as-of date, tanpa syarat lain):")
    regime_stats = []
    for regime in ALL_REGIMES:
        sub = df[df["regime"] == regime]
        n, wr, lb = _stat(sub)
        regime_stats.append((regime, n, wr, lb))
        logger.info("  %-15s n=%-6d win_rate=%.1f%%  wilson_lb=%.1f%%  (%+.1f%% vs null LB)",
                    regime, n, wr * 100, lb * 100, (lb - lb0) * 100)
    regime_stats.sort(key=lambda t: t[3], reverse=True)
    best_regimes = [r for r, n, wr, lb in regime_stats[:2] if lb > lb0]
    logger.info("-> Regime dengan Wilson LB tertinggi (di atas null): %s", best_regimes or "(TIDAK ADA yang mengalahkan null)")

    if not best_regimes:
        logger.info("Tidak ada satu regime pun yang mengalahkan baseline acak untuk target ini -- berhenti di sini.")
        return

    logger.info("=" * 100)
    logger.info("STAGE B -- di dalam regime %s, kondisi teknikal mana yang memisahkan menang/kalah:", best_regimes)
    logger.info("=" * 100)
    base = df[df["regime"].isin(best_regimes)]
    n_base, wr_base, lb_base = _stat(base)
    logger.info("  (regime %s SAJA, tanpa syarat lain): n=%-6d win_rate=%.1f%% wilson_lb=%.1f%%", best_regimes, n_base, wr_base * 100, lb_base * 100)

    conds = {
        "macd_hist_slope_3d>0":        base["macd_hist_slope_3d"] > 0,
        "macd_hist>0 (di atas nol)":   base["macd_hist"] > 0,
        "cmf_20<0 (distribusi)":       base["cmf_20"] < 0,
        "cmf_20>0 (akumulasi)":        base["cmf_20"] > 0,
        "rvol_20>=1.0":                base["rvol_20"] >= 1.0,
        "rvol_20>=1.2":                base["rvol_20"] >= 1.2,
        "rsi_14 30-50":                base["rsi_14"].between(30, 50),
        "rsi_14 45-65":                base["rsi_14"].between(45, 65),
        "ema9>sma20":                  base["ema_9"] > base["sma_20"],
        "sma20>sma50":                 base["sma_20"] > base["sma_50"],
        "adx_14>=20 (tren kuat)":      base["adx_14"] >= 20,
        "relative_strength_20d>0":     base["relative_strength_20d_pct"] > 0,
        "obv_slope_5d>0":              base["obv_slope_5d"] > 0,
        "higher_low_20d":              base["higher_low_20d"] == 1,
        "dist_to_resistance>10%":      base["distance_to_resistance_pct"] > 10,
    }
    results = []
    for name, mask in conds.items():
        sub = base[mask.fillna(False)]
        n, wr, lb = _stat(sub)
        results.append((name, n, wr, lb))
    results.sort(key=lambda t: t[3], reverse=True)
    for name, n, wr, lb in results:
        logger.info("  %-28s n=%-6d win_rate=%.1f%%  wilson_lb=%.1f%%  (%+.1f%% vs regime-only LB)",
                    name, n, wr * 100, lb * 100, (lb - lb_base) * 100)

    logger.info("-" * 100)
    logger.info("Top-3 kondisi tunggal di atas dikombinasikan (AND):")
    top3 = [name for name, n, wr, lb in results[:3]]
    combo_mask = np.logical_and.reduce([conds[name].fillna(False) for name in top3])
    n, wr, lb = _stat(base[combo_mask])
    logger.info("  %s -> n=%-6d win_rate=%.1f%% wilson_lb=%.1f%%", " AND ".join(top3), n, wr * 100, lb * 100)


if __name__ == "__main__":
    run()
