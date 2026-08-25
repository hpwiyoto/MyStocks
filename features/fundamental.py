"""Fundamental snapshot + relative-strength-vs-IHSG, both from yfinance.

Data-quality note found during live testing (2026-08-24, BBCA.JK): yfinance's
`dividendYield` field is on a 0-100 PERCENTAGE scale (e.g. 5.91 means 5.91%),
NOT the 0-1 fraction scale that `trailingAnnualDividendYield` uses (0.0436 =
4.36%) — confirmed by cross-checking against dividendRate/price manually.
`dividend_yield` below stores the percentage-scale value as-is; don't
multiply it by 100 again downstream.

Second data-quality bug found (2026-08-25, ADRO.JK, flagged by the user
cross-checking against Stockbit): `priceToBook` came out as 15470 instead of
a sane ~0.9. Root cause traced precisely: ADRO/Alamtri Resources reports its
financial statements in USD (`financialCurrency=USD`, an export-oriented coal
company) while the stock trades in IDR (`currency=IDR`) — yfinance's
`bookValue` is left in USD but never currency-converted before being divided
into the IDR share price. Confirmed `trailingEps`/`trailingPE` are NOT
affected (EPS is correctly IDR-localized), so only `price_to_book` needs
correction. Fix: when `financialCurrency != currency`, divide the raw
priceToBook by the live FX rate — verified: 15470.588 / 17705 (USDIDR that
day) = 0.874, matching the user's Stockbit reference of 0.91 to within normal
snapshot-timing variance.
"""
import pandas as pd

from pipeline.logging_config import get_logger
from pipeline.yfinance_source import fetch_full_info

logger = get_logger("features.fundamental")

IHSG_SYMBOL = "^JKSE"
RELATIVE_STRENGTH_WINDOW = 20

FUNDAMENTAL_FIELD_MAP = {
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "peg_ratio": "pegRatio",
    "return_on_equity": "returnOnEquity",
    "return_on_assets": "returnOnAssets",
    "profit_margins": "profitMargins",
    "operating_margins": "operatingMargins",
    "dividend_yield": "dividendYield",  # percentage scale, see module docstring
    "payout_ratio": "payoutRatio",
    "market_cap": "marketCap",
    "held_percent_insiders": "heldPercentInsiders",
    "held_percent_institutions": "heldPercentInstitutions",
    "recommendation_mean": "recommendationMean",
    "target_mean_price": "targetMeanPrice",
}


def _fx_corrected_price_to_book(raw_pbv, financial_currency, trading_currency):
    """See module docstring for the ADRO.JK case this guards against."""
    if raw_pbv is None:
        return None
    if not financial_currency or not trading_currency or financial_currency == trading_currency:
        return raw_pbv

    fx_symbol = f"{financial_currency}{trading_currency}=X"
    fx_info = fetch_full_info(fx_symbol)
    rate = fx_info.get("regularMarketPrice") if fx_info else None
    if not rate:
        logger.warning(
            "price_to_book currency mismatch (%s vs %s) but FX rate %s unavailable "
            "-- storing None instead of an uncorrected/wrong ratio",
            financial_currency, trading_currency, fx_symbol,
        )
        return None
    return raw_pbv / rate


def fetch_fundamental_snapshot(symbol: str) -> dict:
    info = fetch_full_info(symbol)
    if not info:
        return {k: None for k in list(FUNDAMENTAL_FIELD_MAP) + ["analyst_upside_pct"]}

    result = {our_key: info.get(yf_key) for our_key, yf_key in FUNDAMENTAL_FIELD_MAP.items()}
    result["price_to_book"] = _fx_corrected_price_to_book(
        result.get("price_to_book"), info.get("financialCurrency"), info.get("currency")
    )

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    target = result.get("target_mean_price")
    if current_price and target:
        result["analyst_upside_pct"] = (target - current_price) / current_price * 100
    else:
        result["analyst_upside_pct"] = None
    return result


def compute_relative_strength(
    stock_close: pd.Series, index_close: pd.Series, window: int = RELATIVE_STRENGTH_WINDOW
) -> float | None:
    """Stock's trailing `window`-day return minus IHSG's, as of the latest
    date both series have in common. None if there isn't enough overlapping
    history yet."""
    aligned = pd.concat(
        [stock_close.rename("stock"), index_close.rename("index")], axis=1, join="inner"
    ).dropna()
    if len(aligned) < window + 1:
        return None

    stock_return = aligned["stock"].iloc[-1] / aligned["stock"].iloc[-1 - window] - 1
    index_return = aligned["index"].iloc[-1] / aligned["index"].iloc[-1 - window] - 1
    return float((stock_return - index_return) * 100)
