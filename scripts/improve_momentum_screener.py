"""Two experiments to raise the Momentum Screener's performance, run
together:

1. ABLATION of the shipped validated_signal rule's 5 conditions
   (regime=bottoming, macd_hist_slope_3d>0, cmf_20<0, rvol_20>=0.8,
   close>=AVWAP-from-50d-low). For each condition: drop it, keep the
   other 4, and measure win_rate / Wilson LB / n. A condition whose
   removal LOSES little edge but ADDS a lot of coverage is dead weight
   worth dropping (the shipped rule fires only ~523 times in 5 years, so
   coverage matters). Each condition alone is also reported for context.

2. RANKING KEY search WITHIN validated-signal instances. The page's
   sort chain past validated_signal (divergence_tier -> regime_priority
   -> rsi_pivot_distance -> ...) is unproven, and scripts/
   backtest_top2_screeners.py showed the top-2 dragged well below the
   rule's own 42.4%. For each candidate feature, split the validated
   instances at their median and compare the better-half vs worse-half
   win rate -- a feature where one half clearly beats the other is a real
   ranking signal that could replace the heuristic tie-break chain. Also
   done per-as-of-date (rank by the feature, take the top pick vs a
   random pick) to mimic the page's actual "show the best one first".

Reuses scripts/test_strategy_6_criteria.build_dataset() (same no-lookahead
5-year dataset that already computes close_above_avwap / dist_to_low50_pct
/ divergence / etc.), exactly as scripts/backtest_triple_intersection.py
does -- no methodology drift.

Usage:
    python -m scripts.improve_momentum_screener
"""
import numpy as np
import pandas as pd

from features.momentum_screener import VALIDATED_RVOL_THRESHOLD
from pipeline.logging_config import get_logger
from scripts.search_momentum_rules import wilson_lower_bound
from scripts.test_strategy_6_criteria import build_dataset

logger = get_logger("scripts.improve_momentum_screener")


def _stat(sub: pd.DataFrame) -> tuple[int, float, float]:
    n = len(sub)
    w = int(sub["outcome"].sum())
    return n, (w / n if n else float("nan")), (wilson_lower_bound(w, n) if n else 0.0)


def run():
    df = build_dataset()
    null_rate = df["outcome"].mean()
    logger.info("=" * 92)
    logger.info("NULL BASELINE: n=%d win_rate=%.4f", len(df), null_rate)
    logger.info("=" * 92)

    conds = {
        "regime=bottoming":       df["regime"] == "bottoming",
        "macd_hist_slope_3d>0":   df["macd_hist_slope_3d"] > 0,
        "cmf_20<0":               df["cmf_20"] < 0,
        f"rvol_20>={VALIDATED_RVOL_THRESHOLD}": df["rvol_20"] >= VALIDATED_RVOL_THRESHOLD,
        "close>=AVWAP(50d-low)":  df["close_above_avwap"] == True,  # noqa: E712
    }
    full = np.logical_and.reduce(list(conds.values()))
    fn, fwr, flb = _stat(df[full])
    logger.info("SHIPPED FULL RULE (all 5): n=%d win_rate=%.4f wilson_lb=%.4f", fn, fwr, flb)

    logger.info("-" * 92)
    logger.info("ABLATION -- drop ONE condition, keep the other 4:")
    for name in conds:
        mask = np.logical_and.reduce([m for k, m in conds.items() if k != name])
        n, wr, lb = _stat(df[mask])
        logger.info("  drop [%-22s] -> n=%-6d win_rate=%.4f wilson_lb=%.4f  (n x%.1f vs full, LB %+.4f)",
                    name, n, wr, lb, n / fn if fn else float("nan"), lb - flb)

    logger.info("-" * 92)
    logger.info("EACH CONDITION ALONE (context):")
    for name, mask in conds.items():
        n, wr, lb = _stat(df[mask])
        logger.info("  [%-22s] -> n=%-6d win_rate=%.4f wilson_lb=%.4f", name, n, wr, lb)

    # -----------------------------------------------------------------
    val = df[full].copy()
    logger.info("=" * 92)
    logger.info("RANKING KEY SEARCH -- within the %d validated-signal instances (base win_rate %.4f)",
                len(val), val["outcome"].mean())
    logger.info("=" * 92)

    # (column, human label). "better" direction is discovered, not assumed.
    candidates = [
        ("rsi_14", "RSI(14)"),
        ("dist_to_low50_pct", "jarak ke low 50d (%)"),
        ("rvol_20", "RVOL(20)"),
        ("macd_hist_slope_3d", "slope MACD hist 3d"),
        ("cmf_20", "CMF(20)"),
        ("cmf_slope_5d", "slope CMF 5d"),
        ("divergence_tier", "divergence tier (0=ganda)"),
        ("divergence_age_days", "umur divergence (hari)"),
    ]
    logger.info("Median split: 'low half' vs 'high half' of each feature, win rate of each:")
    for col, label in candidates:
        s = val[val[col].notna()]
        if len(s) < 40:
            logger.info("  %-30s -- n=%d too small, skipped", label, len(s))
            continue
        med = s[col].median()
        lo, hi = s[s[col] <= med], s[s[col] > med]
        ln, lwr, llb = _stat(lo)
        hn, hwr, hlb = _stat(hi)
        gap = hwr - lwr
        logger.info("  %-30s med=%8.3f | low-half n=%-4d wr=%.4f (LB %.4f) | high-half n=%-4d wr=%.4f (LB %.4f) | gap(hi-lo)=%+.4f",
                    label, med, ln, lwr, llb, hn, hwr, hlb, gap)

    # Per-as-of-date: rank validated instances by the feature, take the #1,
    # compare to the mean outcome of ALL validated instances that day (a
    # "pick one at random" baseline). Averaged over as-of dates.
    logger.info("-" * 92)
    logger.info("Per-day: rank validated hits by feature, take #1 -- vs 'random validated hit that day':")
    day_groups = list(val.groupby("as_of"))
    rand_baseline = np.mean([g["outcome"].mean() for _, g in day_groups if len(g) >= 1])
    logger.info("  random-validated-hit-of-the-day baseline: %.4f (%d days with >=1 validated hit)",
                rand_baseline, sum(1 for _, g in day_groups if len(g) >= 1))
    for col, label in candidates:
        for direction in ("asc", "desc"):
            picks = []
            for _, g in day_groups:
                gg = g[g[col].notna()]
                if gg.empty:
                    continue
                gg = gg.sort_values(col, ascending=(direction == "asc"))
                picks.append(gg.iloc[0]["outcome"])
            if len(picks) < 20:
                continue
            wr = float(np.mean(picks))
            lb = wilson_lower_bound(int(np.sum(picks)), len(picks))
            logger.info("  rank by %-28s %-4s -> #1 pick win_rate=%.4f (LB %.4f, n=%d days)  [%+.4f vs random]",
                        label, direction, wr, lb, len(picks), wr - rand_baseline)


if __name__ == "__main__":
    run()
