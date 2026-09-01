"""RapidAPI "Indonesia Stock Exchange (IDX)" data source -- supplies data
yfinance doesn't have (currently: net foreign buy/sell). Optional: every
function here no-ops (or is simply never called) if RAPIDAPI_KEY isn't set,
same convention as scripts/monitor.py's Telegram alerting.

Found and verified by hand this session (RapidAPI's own docs/example
response bodies didn't answer these on their own):
- `getFundachart` (/api/emiten/fundachart) batches MULTIPLE companies in one
  call, unlike every other per-ticker endpoint on this API -- empirically
  capped between 7-9 companies/call (7 confirmed working, 10 confirmed
  failing with 400). This is what makes a full-universe (914-ticker)
  backfill affordable at all.
- item_id=3194 ("Net Foreign Buy / Sell") is the true daily net figure.
  item_id=3218 ("Foreign Flow") looks similar but is a CUMULATIVE running
  sum -- confirmed by differencing 3218's consecutive values and matching
  them almost exactly to 3194's. Use 3194 directly; no `.diff()` needed.
- The plan's "1 request per second" isn't a formality: 3 back-to-back
  calls with no delay drew a real 429 mid-session.

The free "Basic" RapidAPI plan is 500 requests/month with a $0.01/request
overage past that -- reserve_requests() is a hard, DB-persisted pre-flight
gate (not a log message) specifically so nothing in this module can run
that up by accident, per the user's explicit requirement.

Usage:
    from pipeline.idx_rapidapi_source import fetch_foreign_flow_all
    flows = fetch_foreign_flow_all(["BBCA", "TLKM", ...], timeframe="1y")
    # {code: DataFrame(date, value)}, value in Rupiah (net foreign buy if
    # positive, net foreign sell if negative)
"""
import datetime as dt
import os
import time

import pandas as pd
import requests
from sqlalchemy import insert, select, update

from pipeline.db import api_usage, get_engine
from pipeline.logging_config import get_logger
from pipeline.yfinance_source import with_retry

logger = get_logger("pipeline.idx_rapidapi")

RAPIDAPI_HOST = "indonesia-stock-exchange-idx.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

PROVIDER = "rapidapi_idx"
# Real cap is 500/month (Basic/free plan) -- deliberately budgeting to 480,
# not 500, so a batch that's already reserved-but-not-yet-executed when the
# month rolls over, or a slightly-off count somewhere, never actually tips
# into the $0.01/request overage. This number is a safety margin, not the
# plan's real limit.
DEFAULT_MONTHLY_BUDGET = 480

FOREIGN_FLOW_ITEM_ID = 3194  # "Net Foreign Buy / Sell" -- see module docstring
BATCH_SIZE = 7  # confirmed working; 10 confirmed failing (400)
PACING_SECONDS = 1.1  # confirmed real limit is 1 req/sec; small margin over it
REQUEST_TIMEOUT = 15


def reserve_requests(n: int, provider: str = PROVIDER, budget: int = DEFAULT_MONTHLY_BUDGET) -> bool:
    """Atomically check-and-increment this calendar month's usage counter.
    Returns True and commits the increment only if it would stay within
    `budget`; returns False (no write) otherwise. Callers MUST call this
    BEFORE making the real HTTP call(s) it's reserving for, not after --
    that's the difference between a pre-flight gate and a log message."""
    period = dt.date.today().strftime("%Y-%m")
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            select(api_usage.c.request_count)
            .where(api_usage.c.provider == provider, api_usage.c.billing_period == period)
        ).first()
        current = row[0] if row else 0
        if current + n > budget:
            logger.warning(
                "%s: request budget for %s would be exceeded (%d + %d > %d) -- refusing",
                provider, period, current, n, budget,
            )
            return False
        if row is None:
            conn.execute(insert(api_usage).values(provider=provider, billing_period=period, request_count=n))
        else:
            conn.execute(
                update(api_usage)
                .where(api_usage.c.provider == provider, api_usage.c.billing_period == period)
                .values(request_count=current + n)
            )
    return True


def fetch_fundachart(companies: list[str], item_id: int, timeframe: str = "1y") -> dict[str, pd.DataFrame]:
    """One API call for up to BATCH_SIZE companies at once. Returns
    {stock_code: DataFrame(date, value)}, one row per trading day in
    `timeframe`. Raises after retries on a genuine failure -- callers that
    want per-chunk resilience (e.g. fetch_foreign_flow_all) should catch
    around this, not the other way around."""
    if not RAPIDAPI_KEY:
        logger.warning("RAPIDAPI_KEY not set, skipping fundachart fetch")
        return {}
    if len(companies) > BATCH_SIZE:
        raise ValueError(f"fetch_fundachart: {len(companies)} companies exceeds the confirmed batch cap of {BATCH_SIZE}")

    def _fetch():
        resp = requests.get(
            f"{BASE_URL}/api/emiten/fundachart",
            params={"companies": ",".join(companies), "timeframe": timeframe, "item": item_id},
            headers={"x-rapidapi-host": RAPIDAPI_HOST, "x-rapidapi-key": RAPIDAPI_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    payload = with_retry(_fetch, f"fetch_fundachart({companies}, item={item_id}, timeframe={timeframe})")

    result = {}
    for company in payload.get("data", []):
        code = company.get("company_name")
        ratios = company.get("ratios") or []
        if not code or not ratios:
            continue
        chart_data = ratios[0].get("chart_data") or []
        if not chart_data:
            continue
        df = pd.DataFrame(chart_data)[["formated_date", "value"]].rename(columns={"formated_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        result[code] = df
    return result


def fetch_foreign_flow_all(codes: list[str], timeframe: str = "1y", budget: int = DEFAULT_MONTHLY_BUDGET) -> dict[str, pd.DataFrame]:
    """Chunks `codes` into BATCH_SIZE groups and fetches net foreign
    buy/sell (item_id=FOREIGN_FLOW_ITEM_ID) for each, pacing calls to
    respect the confirmed 1 req/sec limit. Stops early -- logging clearly,
    not silently -- the moment reserve_requests() says the monthly budget
    would be exceeded, rather than partially succeeding without saying so.
    A chunk whose HTTP call fails (after with_retry's own retries) is
    logged and skipped, not fatal to the rest of the run. Returns
    {stock_code: DataFrame(date, value)} for every code that succeeded."""
    if not RAPIDAPI_KEY:
        logger.warning("RAPIDAPI_KEY not set, skipping foreign-flow fetch for %d tickers", len(codes))
        return {}

    results: dict[str, pd.DataFrame] = {}
    chunks = [codes[i:i + BATCH_SIZE] for i in range(0, len(codes), BATCH_SIZE)]
    for i, chunk in enumerate(chunks):
        if not reserve_requests(1, budget=budget):
            logger.warning(
                "Stopping foreign-flow fetch after %d/%d batches (%d/%d tickers fetched) -- monthly budget reached",
                i, len(chunks), len(results), len(codes),
            )
            break
        try:
            chunk_result = fetch_fundachart(chunk, FOREIGN_FLOW_ITEM_ID, timeframe)
        except Exception as exc:
            logger.error("fundachart fetch failed for chunk %s: %s", chunk, exc)
            chunk_result = {}
        results.update(chunk_result)
        missing = sorted(set(chunk) - set(chunk_result))
        if missing:
            logger.warning("No fundachart data returned for: %s", missing)
        if i < len(chunks) - 1:
            time.sleep(PACING_SECONDS)

    logger.info("fetch_foreign_flow_all: %d/%d tickers fetched", len(results), len(codes))
    return results
