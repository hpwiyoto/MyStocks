"""Import foreign buy/sell/frequency/value from a manually downloaded IDX CSV.

Manual step (you, as a human, do this in a browser — not automated, since
IDX's site blocks automated access):
  1. Buka https://www.idx.co.id > Data Pasar > Ringkasan Perdagangan.
  2. Pilih tanggal, klik Download/Export ke CSV.
  3. Jalankan: python -m pipeline.import_foreign_flow path/to/file.csv

Column names in IDX's export can vary by report/locale. This script normalizes
headers (case/whitespace-insensitive) and accepts several known synonyms — see
COLUMN_ALIASES below. If your file's columns aren't recognized, the script
prints exactly what it found and what's missing so the mapping can be fixed in
one pass instead of silently importing wrong data.
"""
import sys

import pandas as pd
from sqlalchemy.dialects.mysql import insert as mysql_insert

from pipeline.db import foreign_flow, get_engine, init_schema
from pipeline.logging_config import get_logger

logger = get_logger("pipeline.import_foreign_flow")

SOURCE = "idx_manual_csv"

# canonical_field -> accepted header spellings (lowercased, whitespace-stripped)
COLUMN_ALIASES = {
    "stock_code": ["stockcode", "kode saham", "kode", "code"],
    "date": ["date", "tanggal"],
    "foreign_buy": ["foreignbuy", "foreign buy", "asing beli"],
    "foreign_sell": ["foreignsell", "foreign sell", "asing jual"],
    "frequency": ["frequency", "frekuensi"],
    "value": ["value", "nilai"],
}


def _normalize(col: str) -> str:
    return str(col).strip().lower()


def _safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _map_columns(df: pd.DataFrame) -> dict:
    normalized = {_normalize(c): c for c in df.columns}
    mapping = {}
    missing = []
    for canonical, aliases in COLUMN_ALIASES.items():
        found = next((normalized[a] for a in aliases if a in normalized), None)
        if found is None:
            missing.append(canonical)
        else:
            mapping[canonical] = found

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di CSV: {missing}. "
            f"Kolom yang tersedia di file: {list(df.columns)}. "
            f"Tambahkan alias yang sesuai di COLUMN_ALIASES pada pipeline/import_foreign_flow.py."
        )
    return mapping


def import_csv(path: str) -> int:
    df = pd.read_csv(path)
    if df.empty:
        logger.warning("%s: file kosong, tidak ada yang diimpor", path)
        return 0

    mapping = _map_columns(df)
    rows = []
    for _, row in df.iterrows():
        code = str(row[mapping["stock_code"]]).strip().upper()
        rows.append(
            {
                "stock_code": code,
                "date": pd.to_datetime(row[mapping["date"]]).date(),
                "foreign_buy": _safe_int(row[mapping["foreign_buy"]]),
                "foreign_sell": _safe_int(row[mapping["foreign_sell"]]),
                "frequency": _safe_int(row[mapping["frequency"]]),
                "value": _safe_int(row[mapping["value"]]),
                "source": SOURCE,
            }
        )

    engine = get_engine()
    init_schema(engine)
    with engine.begin() as conn:
        stmt = mysql_insert(foreign_flow).values(rows)
        update_cols = {c: stmt.inserted[c] for c in (
            "foreign_buy", "foreign_sell", "frequency", "value"
        )}
        stmt = stmt.on_duplicate_key_update(**update_cols)
        conn.execute(stmt)

    logger.info("%s: berhasil impor %d baris ke foreign_flow", path, len(rows))
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.import_foreign_flow path/to/file.csv")
        sys.exit(1)
    import_csv(sys.argv[1])
