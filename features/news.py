"""Headline-only news lookup for the Detail Saham UI panel -- NOT a model
feature (see chat: yfinance's own `.news` was tested and is too sparse/stale
for Indonesian small-caps -- 0 items for BBCA/ASII even -- and any feed only
has a few days/weeks of lookback, nowhere near the ~3-year walk-forward
training window, the same point-in-time-history problem already documented
for fundamentals in build_features._merge_fundamental_features).

Uses Google News' RSS search endpoint, which returns fresh, Indonesian-
language, relevant results (verified: BBCA search returned same-day
sekuritas/analyst coverage) -- Google's own feed copyright notice permits
exactly this use ("solely for the purpose of rendering Google News results
within a personal feed reader for personal, non-commercial use"). Only
headline + source + date + a link back to the original article is shown;
this module never fetches or scrapes the linked article's own page, which
would raise separate ToS questions for each destination site.
"""
import datetime as dt
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pipeline.logging_config import get_logger

logger = get_logger("features.news")

RSS_BASE = "https://news.google.com/rss/search"
REQUEST_TIMEOUT = 8
MAX_ITEMS = 6
# A single retry, short backoff -- this is a synchronous call inside a
# Streamlit page render (the user is waiting), unlike pipeline.yfinance_source's
# 3-attempt/5s-step retry for a background ingest job. Added after a real
# one-off failure (WinError 10054, connection reset) turned into a page
# showing "no news" for a ticker that had news moments before AND after --
# app/data.py's load_news caches whatever this returns for 30 minutes, so a
# transient blip that isn't smoothed over here stays visibly wrong for a
# while, not just for that one page load.
MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2


def fetch_news_headlines(query: str, max_items: int = MAX_ITEMS) -> list[dict]:
    """query: usually "{ticker_code} saham" -- plain company/ticker name
    search, Indonesian-locale results. Returns [] on any failure (network,
    parse, no results) rather than raising -- this is a best-effort display
    panel, never something a page should break over."""
    params = {"q": query, "hl": "id", "gl": "ID", "ceid": "ID:id"}
    url = f"{RSS_BASE}?{urllib.parse.urlencode(params)}"
    root = None
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            break
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    if root is None:
        logger.warning("news fetch failed for %r after %d attempt(s): %s", query, MAX_ATTEMPTS, last_error)
        return []

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_raw = item.findtext("pubDate")
        pub_date = None
        if pub_date_raw:
            try:
                pub_date = dt.datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                pub_date = None
        # Google's title format is "Headline - Source Name" -- split off the
        # source for display rather than showing the raw combined string.
        if " - " in title:
            headline, source = title.rsplit(" - ", 1)
        else:
            headline, source = title, None
        if not headline:
            continue
        items.append({"title": headline, "source": source, "link": link, "pub_date": pub_date})
    return items
