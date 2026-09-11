"""Tune the +X%/-Y% magnitude for the Turnaround v2 target now that the
regime/condition side is settled (scripts/search_turnaround_v2_target.py,
3 rounds): overextended/bullish regime, best single condition
price_vs_vwap20_pct>5 (wide net, LB 37.9%) or the tighter 3-way combo
ret_10d_atr_norm>1.0 AND rvol_20>=1.2 AND bb_width_change_5d>0 (LB 38.0%,
narrower). +15%/-7.5% was a reasoned STARTING guess (Swing's 5%/2.5%
ratio scaled by sqrt(time) for a ~6x longer horizon), never actually
tuned -- this does that.

Loads the panel and computes each ticker's forward HIGH/LOW windows ONCE
(horizon fixed at 60 trading days -- "maksimum 3 bulan" is the user's own
ceiling, not something to relax) since that's the expensive, magnitude-
INDEPENDENT part; only the target/stop threshold comparison (cheap) is
redone per magnitude in the grid. Grid keeps Swing's 2:1 reward:risk ratio
fixed (target_pct, stop_pct=target_pct/2) and sweeps the target size:
10/15/20/25/30%.

For each magnitude: null (all regimes), regime-only (overextended+
bullish), WIDE rule (regime + price_vs_vwap20_pct>5), TIGHT combo (regime
+ ret_10d_atr_norm>1.0 & rvol_20>=1.2 & bb_width_change_5d>0) -- n/
win_rate/wilson_lb for each, so the magnitude that maximizes the RULE's
lift over null (not just its raw win rate, which any magnitude can
inflate by loosening the target) is visible directly.

Usage:
    python -m scripts.tune_turnaround_v2_magnitude
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import load_full_panel, momentum_extra_indicators
from scripts.search_momentum_rules import wilson_lower_bound

logger = get_logger("scripts.tune_turnaround_v2_magnitude")

HORIZON = 60
WARMUP = 60
AS_OF_STRIDE = 10
MAGNITUDE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]  # target_pct; stop_pct = target_pct / 2 (Swing's 2:1 ratio)
CANDIDATE_REGIMES = {"overextended", "bullish"}


def precompute() -> list[dict]:
    """One entry per ticker: static columns (feature_daily at each as-of
    index) + the forward high/low windows (magnitude-independent)."""
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    tickers = []
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= WARMUP + HORIZON:
            continue
        g = momentum_extra_indicators(g)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        valid_n = n - HORIZON
        idxs = np.arange(WARMUP, valid_n)
        if len(idxs) == 0:
            continue
        fwd_high = sliding_window_view(high[1:], HORIZON)[idxs]  # (len(idxs), HORIZON)
        fwd_low = sliding_window_view(low[1:], HORIZON)[idxs]
        entry = close[idxs]
        static = g.iloc[idxs].reset_index(drop=True)
        tickers.append({"code": code, "static": static, "entry": entry, "fwd_high": fwd_high, "fwd_low": fwd_low})
    logger.info("Precomputed forward windows for %d tickers", len(tickers))
    return tickers


def dataset_for_magnitude(tickers: list[dict], target_pct: float, stop_pct: float) -> pd.DataFrame:
    rows = []
    for t in tickers:
        entry = t["entry"]
        target_hit = t["fwd_high"] >= (entry[:, None] * (1 + target_pct))
        stop_hit = t["fwd_low"] <= (entry[:, None] * (1 - stop_pct))
        target_any = target_hit.any(axis=1)
        stop_any = stop_hit.any(axis=1)
        first_target_t = np.where(target_any, target_hit.argmax(axis=1), HORIZON)
        first_stop_t = np.where(stop_any, stop_hit.argmax(axis=1), HORIZON)
        resolved = target_any | stop_any
        win = first_target_t < first_stop_t
        outcome = np.where(resolved, win.astype(float), np.nan)
        sub = t["static"].copy()
        sub["outcome"] = outcome
        sub["stock_code"] = t["code"]
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["stock_code", "date"])
    out["_rank"] = out.groupby("stock_code").cumcount()
    out = out[out["_rank"] % AS_OF_STRIDE == 0].drop(columns="_rank")
    return out.dropna(subset=["outcome"])


def _stat(sub: pd.DataFrame) -> tuple[int, float, float]:
    n = len(sub)
    if n == 0:
        return 0, float("nan"), 0.0
    w = int(sub["outcome"].sum())
    return n, w / n, wilson_lower_bound(w, n)


def run():
    tickers = precompute()
    logger.info("=" * 110)
    logger.info("%-16s | %-28s | %-28s | %-28s | %-28s", "target/-stop", "NULL (semua)", "REGIME saja", "WIDE (regime+VWAP>5)", "TIGHT (regime+3 kondisi)")
    logger.info("=" * 110)
    for target_pct in MAGNITUDE_GRID:
        stop_pct = target_pct / 2
        df = dataset_for_magnitude(tickers, target_pct, stop_pct)
        n0, wr0, lb0 = _stat(df)
        regime_mask = df["regime"].isin(CANDIDATE_REGIMES)
        n1, wr1, lb1 = _stat(df[regime_mask])
        wide_mask = regime_mask & (df["price_vs_vwap20_pct"] > 5)
        n2, wr2, lb2 = _stat(df[wide_mask])
        tight_mask = regime_mask & (df["ret_10d_atr_norm"] > 1.0) & (df["rvol_20"] >= 1.2) & (df["bb_width_change_5d"] > 0)
        n3, wr3, lb3 = _stat(df[tight_mask])
        logger.info(
            "+%.0f%%/-%.1f%%   | n=%-6d %4.1f%% (LB%5.1f) | n=%-6d %4.1f%% (LB%5.1f) | n=%-6d %4.1f%% (LB%5.1f) | n=%-6d %4.1f%% (LB%5.1f)",
            target_pct * 100, stop_pct * 100,
            n0, wr0 * 100, lb0 * 100, n1, wr1 * 100, lb1 * 100, n2, wr2 * 100, lb2 * 100, n3, wr3 * 100, lb3 * 100,
        )
        logger.info("   -> lift vs null (LB): regime-saja %+.1fpp | WIDE %+.1fpp | TIGHT %+.1fpp",
                    (lb1 - lb0) * 100, (lb2 - lb0) * 100, (lb3 - lb0) * 100)


if __name__ == "__main__":
    run()
