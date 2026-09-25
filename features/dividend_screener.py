"""Seasonal dividend screener for the Momentum Screener page -- direct
user request: find high-yield dividend payers that HISTORICALLY tend to
go ex-dividend in the next ~1-2 months, as a rough seasonal hint.

IMPORTANT LIMITATION, stated up front rather than buried: there is no
reliable FORWARD-LOOKING dividend calendar available to this project.
Confirmed empirically (2026-09-25): yfinance's `exDividendDate` /
`lastDividendDate` fields report the MOST RECENT PAST ex-date, not an
upcoming announced one (checked BBCA/BBRI/TLKM/ADRO -- every one of
those fields matched exactly the last row of that ticker's own
`.dividends` history, e.g. BBCA's "exDividendDate" was 2026-08-31,
already 3.5 weeks in the past on the day this was checked). IDX's own
site -- the actual source of forward corporate-action calendars -- is
Cloudflare-blocked (see scripts/special_monitoring_board.py's docstring
for that investigation), and the RapidAPI IDX source already integrated
in this project (pipeline/idx_rapidapi_source.py) has no confirmed
dividend-calendar endpoint, and its budget is too tight to speculatively
probe for one.

So: this module can only show WHEN a ticker HAS paid dividends in past
years (from price_history.dividends, already ingested as a side effect
of routine OHLCV fetches -- see pipeline/ingest_price.py) and flag
whether that historical MONTH pattern overlaps the next ~1-2 months --
a seasonal hint based on past behavior, not a confirmed schedule. Actual
cum-dates shift year to year (RUPS timing, corporate decisions) --
always verify against an official announcement/keterbukaan informasi
before acting on this for a real trade.
"""
import datetime as dt

import pandas as pd

# User's own bar: "lumayan besar, minimal diatas 5% atau diatas suku
# bunga bank" -- 5.0% used directly as a fixed floor (no live central-
# bank-rate feed exists in this project; BI's policy rate has hovered
# in a similar range recently, so this is a reasonable proxy -- revisit
# if it moves meaningfully).
DIVIDEND_YIELD_MIN_PCT = 5.0
# User follow-up ("gak harus 2 bulan, minimal 1-2 bulan lah") -- current
# month + this many months ahead counts as "upcoming".
LOOKAHEAD_MONTHS = 2
MIN_YEARS_OF_HISTORY = 1  # at least one full realized payment on record


def historical_dividend_months(dividend_dates: list) -> list[int]:
    """Distinct calendar months (1-12) this ticker has paid a dividend in
    across all available history, deduplicated regardless of year --
    e.g. a company that paid in March 2024, March 2025, and August 2025
    returns [3, 8]."""
    return sorted({d.month for d in dividend_dates})


def is_seasonally_upcoming(dividend_months: list[int], as_of: dt.date, lookahead_months: int = LOOKAHEAD_MONTHS) -> bool:
    """True if the ticker has historically paid in the current calendar
    month or any of the next `lookahead_months` months (wrapping across
    a year boundary, e.g. Nov + 2 months reaches Dec and Jan) -- a MONTH
    match, not an exact date -- see module docstring for why an exact
    forward date isn't available at all."""
    target_months = {((as_of.month - 1 + i) % 12) + 1 for i in range(lookahead_months + 1)}
    return bool(target_months & set(dividend_months))


MONTH_NAMES_ID = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
    7: "Jul", 8: "Agt", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}


def summarize_dividend_screen(panel: pd.DataFrame, as_of: dt.date | None = None) -> pd.DataFrame:
    """panel: long-format rows (stock_code, date, dividends, dividend_yield,
    payout_ratio) from app.data.load_dividend_screener_data -- `date`/
    `dividends` are the REALIZED historical payments (one row per actual
    past payment), `dividend_yield`/`payout_ratio` repeat per ticker
    (same value on every row for that stock_code, taken from its latest
    fundamental snapshot).

    Returns one row per ticker that clears DIVIDEND_YIELD_MIN_PCT AND
    is_seasonally_upcoming, sorted by yield descending -- tickers with no
    realized dividend history at all, or whose yield is missing/below
    the bar, or whose historical payment months don't overlap the
    lookahead window, are dropped entirely (this is a screener, not a
    ranking of everything).
    """
    as_of = as_of or dt.date.today()
    if panel.empty:
        return pd.DataFrame()

    rows = []
    for code, g in panel.groupby("stock_code"):
        g = g.sort_values("date")
        yield_pct = g["dividend_yield"].iloc[-1]
        if pd.isna(yield_pct) or yield_pct < DIVIDEND_YIELD_MIN_PCT:
            continue
        hist = g.dropna(subset=["date", "dividends"])
        hist = hist[hist["dividends"] > 0]
        if hist.empty:
            continue
        dates = [pd.Timestamp(d).date() for d in hist["date"]]
        years = len({d.year for d in dates})
        if years < MIN_YEARS_OF_HISTORY:
            continue
        months = historical_dividend_months(dates)
        if not is_seasonally_upcoming(months, as_of):
            continue
        last_date = dates[-1]
        last_amount = float(hist["dividends"].iloc[-1])
        rows.append({
            "stock_code": code,
            "dividend_yield_pct": float(yield_pct),
            "payout_ratio": g["payout_ratio"].iloc[-1],
            "historical_months": months,
            "historical_months_display": ", ".join(MONTH_NAMES_ID[m] for m in months),
            "last_dividend_date": last_date,
            "last_dividend_amount": last_amount,
            "years_of_history": years,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("dividend_yield_pct", ascending=False).reset_index(drop=True)
    return out
