"""Does a Swing BUY signal carry real historical risk of the stock getting
SUSPENDED (trading halted) afterward -- a different, worse failure mode
than a normal stop-loss (a suspension freezes the position; you can't
exit at any price until it's lifted, sometimes for months)?

Prompted by a direct user question after the LUCY case (LUCY itself was
NOT suspended -- still trading normally through 2026-09-11, it just
crashed hard -- so this tests the GENERAL question, not that specific
case).

No suspension-flag data source exists in this project (checked: no
column, no pipeline step tracks it). Proxy used instead: a ticker-
specific gap of >=GAP_THRESHOLD consecutive trading days with NO
price_history row, while the overall market (other tickers) kept
trading normally in that window -- ruling out holidays/weekends, which
already don't produce calendar entries at all. Not perfect (a data-
pipeline miss for one ticker would look identical to a real IDX
suspension), but the only signal available; found gaps are inspected
by hand below, not just counted.

Also flags the deeper methodological point this surfaces: the triple-
barrier win/loss labels used throughout this project's backtests
(target OR stop touched within the horizon) silently EXCLUDE an
instance where NEITHER barrier is touched -- which is exactly what a
frozen/suspended price would produce. A real suspension after a BUY
signal would not show up as a loss dragging down Swing's reported
precision; it would just vanish from the sample. This script's finding
is about whether that blind spot has ever actually been exercised in
this data, not a claim that it structurally can't happen.

Usage:
    python -m scripts.check_suspension_risk
"""
import pandas as pd

from pipeline.db import get_engine
from pipeline.logging_config import get_logger

logger = get_logger("scripts.check_suspension_risk")

GAP_THRESHOLD_TRADING_DAYS = 10


def run():
    engine = get_engine()
    df = pd.read_sql("SELECT stock_code, date FROM price_history WHERE source_provider = 'yfinance'", engine)
    df["date"] = pd.to_datetime(df["date"])
    calendar = sorted(df["date"].unique())
    cal_idx = {d: i for i, d in enumerate(calendar)}
    last_idx = len(calendar) - 1
    logger.info("Market calendar: %d trading days, %s -> %s", len(calendar), calendar[0].date(), calendar[-1].date())

    gaps = []
    for code, g in df.groupby("stock_code"):
        idxs = sorted(cal_idx[d] for d in g["date"].unique())
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = b - a - 1
            if gap >= GAP_THRESHOLD_TRADING_DAYS:
                gaps.append({"stock_code": code, "gap_start_after": calendar[a], "resumes_at": calendar[b], "gap_trading_days": gap})
        if last_idx - idxs[-1] >= GAP_THRESHOLD_TRADING_DAYS:
            gaps.append({"stock_code": code, "gap_start_after": calendar[idxs[-1]], "resumes_at": None, "gap_trading_days": last_idx - idxs[-1]})

    gdf = pd.DataFrame(gaps)
    logger.info("=" * 90)
    logger.info("Suspension-like gaps found (>=%d trading days, out of %d tickers / %d trading days of history):",
                GAP_THRESHOLD_TRADING_DAYS, df["stock_code"].nunique(), len(calendar))
    logger.info("  %d gaps across %d unique tickers", len(gdf), gdf["stock_code"].nunique() if not gdf.empty else 0)
    logger.info("=" * 90)
    if gdf.empty:
        logger.info("None found.")
        return
    logger.info("\n%s", gdf.sort_values("gap_trading_days", ascending=False).to_string(index=False))

    logger.info("-" * 90)
    logger.info("Volume in the 10 sessions right before each affected ticker's FIRST gap (checking whether it looks")
    logger.info("like a real halt after active trading, or a chronically-dormant name that was never BUY-eligible):")
    for code in gdf["stock_code"].unique():
        first_gap_start = gdf[gdf["stock_code"] == code]["gap_start_after"].min()
        window = df[(df["stock_code"] == code) & (df["date"] <= first_gap_start)].sort_values("date").tail(10)
        vol = pd.read_sql(
            f"SELECT date, close, volume FROM price_history WHERE stock_code = '{code}' AND source_provider = 'yfinance' "
            f"AND date <= '{first_gap_start.date()}' ORDER BY date DESC LIMIT 10",
            engine,
        )
        avg_vol = vol["volume"].mean()
        logger.info("  %s: avg volume/day in the 10 sessions before its first gap = %.0f shares", code, avg_vol)


if __name__ == "__main__":
    run()
