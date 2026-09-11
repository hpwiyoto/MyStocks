"""Does a NEW kind of call -- "bottoming/sideways stock showing early signs
of waking up" -- have real historical odds of a big move (+20%/+30%/+50%/
a bagger +100%) within a MAXIMUM of 3 trading months (60 trading days)?

Explicitly a DIFFERENT question from the existing Turnaround page, not a
replacement for it: Turnaround targets "will this bearish/bottoming stock
reach early_reversal/bullish and HOLD there >=20 trading days, within 6
months" -- a regime-durability question. This tests a price-magnitude
question instead ("will it actually pop big"), over a shorter max horizon,
starting from a wider regime set (bottoming AND sideways, not just
bearish/bottoming), using price action alone (no held-regime requirement).

Methodology matches every other rule-search script in this project:
- Universe: (ticker, as-of date) pairs where regime is bottoming or
  sideways that day (BAD_REGIMES most people would call "not yet obviously
  good", but sideways here too since the user explicitly asked for it).
- Outcome: did the stock's HIGH touch >=X% above the as-of close at ANY
  point in the next 60 trading days (not just at day 60 -- "maksimum 3
  bulan" means "within", so the running maximum is the right measure, same
  logic as touch_target_no_stop in test_stochrsi_early_reversal_rule.py).
  Tested at X in {20%, 30%, 50%, 100% (a "bagger")}.
- Three baselines to separate "the regime filter alone" from "the extra
  early-wake-up signal on top of it":
    1. UNCONDITIONAL null -- any regime, any as-of date.
    2. REGIME-ONLY -- bottoming/sideways, no other filter.
    3. FULL RULE -- bottoming/sideways + all 4 candidate "waking up"
       conditions (MACD histogram turning up, volume above its own
       average, EMA9 back above SMA20, RSI recovering through the
       midzone) -- plus an ablation dropping each one at a time, same
       structure as scripts/improve_momentum_screener.py's experiment 1.
- Wilson lower bound (95%) on every hit rate, not just the point estimate
  -- same overfitting guard used everywhere else here.

As-of dates strided (every AS_OF_STRIDE-th trading day) to keep
overlapping 60-day forward windows from inflating n with near-duplicate
observations, same convention as search_momentum_rules.py.

Usage:
    python -m scripts.test_turnaround_early_bounce
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.logging_config import get_logger
from scripts.backtest_top2_screeners import load_full_panel
from scripts.search_momentum_rules import wilson_lower_bound

logger = get_logger("scripts.test_turnaround_early_bounce")

HORIZON_3M = 60          # trading days -- "maksimum 3 bulan"
WARMUP = 60              # bars of history required before an as-of date is used (indicator stability)
AS_OF_STRIDE = 10
THRESHOLDS = [0.20, 0.30, 0.50, 1.00]
CANDIDATE_REGIMES = {"bottoming", "sideways"}


def build_dataset() -> pd.DataFrame:
    df = load_full_panel()
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n <= WARMUP + HORIZON_3M:
            continue
        high = g["high"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        valid_n = n - HORIZON_3M
        # max_fwd[i] = max(high[i+1 .. i+HORIZON_3M]) -- the running max over
        # the NEXT HORIZON_3M bars, excluding the as-of bar itself.
        max_fwd = sliding_window_view(high[1:], HORIZON_3M).max(axis=1)
        close_at_h = close[HORIZON_3M:n]  # close[i+HORIZON_3M]
        idxs = np.arange(WARMUP, valid_n)
        if len(idxs) == 0:
            continue
        sub = g.iloc[WARMUP:valid_n].copy()
        entry_close = close[idxs]
        sub["max_ret_3m"] = max_fwd[idxs] / entry_close - 1
        sub["close_ret_3m"] = close_at_h[idxs] / entry_close - 1
        sub["stock_code"] = code
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    # Stride the as-of axis per ticker (not globally -- tickers have
    # slightly different trading calendars/history lengths), same idea as
    # every other script here.
    out = out.sort_values(["stock_code", "date"])
    out["_rank"] = out.groupby("stock_code").cumcount()
    out = out[out["_rank"] % AS_OF_STRIDE == 0].drop(columns="_rank")
    logger.info("Dataset: %d (ticker, as-of) rows across %d tickers", len(out), out["stock_code"].nunique())
    return out


def _stat(sub: pd.DataFrame, col: str, thr: float) -> tuple[int, float, float]:
    s = sub[col].dropna()
    n = len(s)
    if n == 0:
        return 0, float("nan"), 0.0
    w = int((s >= thr).sum())
    return n, w / n, wilson_lower_bound(w, n)


def run():
    df = build_dataset()

    logger.info("=" * 100)
    logger.info("HIT RATE per threshold -- 'reached >=X%% at some point within %d trading hari (~3 bulan)'", HORIZON_3M)
    logger.info("=" * 100)

    regime_mask = df["regime"].isin(CANDIDATE_REGIMES)
    conds = {
        "macd_hist_slope_3d>0":  df["macd_hist_slope_3d"] > 0,
        "rvol_20>=1.0":          df["rvol_20"] >= 1.0,
        "ema9>sma20":            df["ema_9"] > df["sma_20"],
        "rsi_recovering(35-65,naik)": df["rsi_14"].between(35, 65) & (df["rsi_slope_3d"] > 0),
    }
    full_mask = regime_mask & np.logical_and.reduce(list(conds.values()))

    for thr in THRESHOLDS:
        n0, wr0, lb0 = _stat(df, "max_ret_3m", thr)
        n1, wr1, lb1 = _stat(df[regime_mask], "max_ret_3m", thr)
        n2, wr2, lb2 = _stat(df[full_mask], "max_ret_3m", thr)
        logger.info("+%.0f%%: null(semua regime) n=%-6d %.1f%% (LB %.1f)  |  bottoming/sideways SAJA n=%-6d %.1f%% (LB %.1f)  |  FULL RULE n=%-5d %.1f%% (LB %.1f)",
                    thr * 100, n0, wr0 * 100, lb0 * 100, n1, wr1 * 100, lb1 * 100, n2, wr2 * 100, lb2 * 100)

    logger.info("-" * 100)
    logger.info("Distribusi return (close_ret_3m = return AKTUAL di hari ke-%d, bukan puncak; max_ret_3m = puncak tertinggi selama window):", HORIZON_3M)
    for label, mask in [("null (semua)", pd.Series(True, index=df.index)), ("bottoming/sideways", regime_mask), ("FULL RULE", full_mask)]:
        sub = df[mask]
        logger.info("  %-20s n=%-6d  median max_ret=%+.1f%%  median close_ret_%dd=%+.1f%%  mean max_ret=%+.1f%%",
                    label, len(sub), sub["max_ret_3m"].median() * 100, HORIZON_3M, sub["close_ret_3m"].median() * 100,
                    sub["max_ret_3m"].mean() * 100)

    logger.info("=" * 100)
    logger.info("ABLATION (di dalam FULL RULE) pada threshold +30%% -- lepas satu kondisi, sisakan yang lain:")
    logger.info("=" * 100)
    n_full, wr_full, lb_full = _stat(df[full_mask], "max_ret_3m", 0.30)
    logger.info("  FULL RULE                          n=%-5d %.1f%% (LB %.1f)", n_full, wr_full * 100, lb_full * 100)
    for name in conds:
        mask = regime_mask & np.logical_and.reduce([m for k, m in conds.items() if k != name])
        n, wr, lb = _stat(df[mask], "max_ret_3m", 0.30)
        logger.info("  drop [%-28s] -> n=%-5d %.1f%% (LB %.1f)  (n x%.1f, LB %+.1f)",
                    name, n, wr * 100, lb * 100, n / n_full if n_full else float("nan"), (lb - lb_full) * 100)
    logger.info("  Masing-masing kondisi SENDIRIAN (dalam regime bottoming/sideways):")
    for name, mask in conds.items():
        n, wr, lb = _stat(df[regime_mask & mask], "max_ret_3m", 0.30)
        logger.info("    [%-28s] -> n=%-5d %.1f%% (LB %.1f)", name, n, wr * 100, lb * 100)


if __name__ == "__main__":
    run()
