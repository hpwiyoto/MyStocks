# Starter list of liquid IDX blue-chip stock codes (without the .JK suffix).
# Extend this list as coverage grows; ingest_price.py upserts each into `stocks`
# on first run using name/sector fetched from yfinance.
SEED_TICKERS = [
    "BBCA",
    "BBRI",
    "BMRI",
    "BBNI",
    "TLKM",
    "ASII",
    "UNVR",
    "ICBP",
    "ANTM",
    "ADRO",
]


def to_yfinance_symbol(code: str) -> str:
    return f"{code}.JK"
