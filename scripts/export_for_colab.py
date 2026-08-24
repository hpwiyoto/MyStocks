"""Export feature_daily + price_history (high/low/close, for label
construction) to a single Parquet file for upload to Google Colab.

Colab can't reach a local/dev MySQL instance over the network, so Fase 3
research loads from this file rather than a live DB connection. Re-run this
whenever you want the Colab notebook to see fresh data.

Usage:
    python -m scripts.export_for_colab [output_path]
"""
import sys

import pandas as pd

from pipeline.db import get_engine
from pipeline.logging_config import get_logger

logger = get_logger("scripts.export_for_colab")

DEFAULT_OUTPUT = "data/export_for_colab.parquet"


def export(output_path: str = DEFAULT_OUTPUT) -> None:
    engine = get_engine()
    features = pd.read_sql("SELECT * FROM feature_daily", engine)
    prices = pd.read_sql(
        "SELECT stock_code, date, high, low, close FROM price_history WHERE source_provider='yfinance'",
        engine,
    )

    features.to_parquet(output_path.replace(".parquet", "_features.parquet"), index=False)
    prices.to_parquet(output_path.replace(".parquet", "_prices.parquet"), index=False)

    logger.info(
        "Exported %d feature_daily rows and %d price_history rows to %s / %s",
        len(features), len(prices),
        output_path.replace(".parquet", "_features.parquet"),
        output_path.replace(".parquet", "_prices.parquet"),
    )


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
