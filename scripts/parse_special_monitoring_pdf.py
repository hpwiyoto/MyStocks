"""Parses IDX's official "Papan Pemantauan Khusus" (Special Monitoring
Board) PDF export into (stock_code, entry_date, exit_date) rows -- the
same PDF format the user hand-provided once already (see
scripts/special_monitoring_board.py's docstring for that history). Lets
the Admin page's PDF upload replace manual transcription going forward.

Each real row in this PDF renders as ONE line of extracted text: a
4-letter uppercase ticker code, the company name, an entry date
("Tanggal Masuk"), and optionally an exit date ("Tanggal Keluar") if the
stock has since left the board. Header/footer/page-number lines never
start with 4 consecutive uppercase letters (IDX tickers always are
exactly that), so they're skipped for free by CODE_RE below rather than
needing an explicit denylist.

Known PDF-extraction quirk (confirmed on the actual PDF that seeded
scripts/special_monitoring_board.py): a stray trailing digit sometimes
gets glued onto a date with no separating space (e.g. "29 Agt 20257").
DATE_RE's exactly-4-digits year group naturally recovers the correct
year from this (\\d{4} only ever consumes 4 characters, so "20257"
yields "2025" with the stray "7" simply left over, unconsumed) without
any special-case code.
"""
import datetime as dt
import io
import re

MONTHS_ID = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Mei": 5, "Jun": 6,
    "Jul": 7, "Agt": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Des": 12,
}
_MONTH_ALTERNATION = "|".join(MONTHS_ID)
CODE_RE = re.compile(r"^([A-Z]{4})\b")
DATE_RE = re.compile(rf"(\d{{1,2}})\s+({_MONTH_ALTERNATION})\s+(\d{{4}})")


def _to_date(day: str, month_abbr: str, year: str) -> dt.date | None:
    try:
        return dt.date(int(year), MONTHS_ID[month_abbr], int(day))
    except ValueError:
        return None


def parse_pdf_bytes(data: bytes) -> tuple[list[dict], list[str]]:
    """Returns (rows, skipped_lines).

    rows: list of {"stock_code", "entry_date", "exit_date"} -- exit_date
    is None if the line had only one date (still active on the board).

    skipped_lines: every line that started with a 4-letter uppercase
    code but couldn't be parsed into at least one valid date -- surfaced
    by the Admin page's preview so a genuinely malformed row is visible
    instead of silently dropped.
    """
    import pdfplumber

    rows: list[dict] = []
    skipped: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                code_match = CODE_RE.match(line)
                if not code_match:
                    continue
                dates = DATE_RE.findall(line)
                if not dates:
                    skipped.append(line)
                    continue
                entry_date = _to_date(*dates[0])
                exit_date = _to_date(*dates[1]) if len(dates) >= 2 else None
                if entry_date is None:
                    skipped.append(line)
                    continue
                rows.append({"stock_code": code_match.group(1), "entry_date": entry_date, "exit_date": exit_date})
    return rows, skipped


def active_tickers(rows: list[dict]) -> set[str]:
    """A code is active if its MOST RECENT row (by entry_date) has no
    exit_date -- handles a code that appears more than once across
    separate monitoring periods (re-entered the board later than an
    earlier, already-closed-out period)."""
    latest_by_code: dict[str, dict] = {}
    for row in rows:
        code = row["stock_code"]
        if code not in latest_by_code or row["entry_date"] > latest_by_code[code]["entry_date"]:
            latest_by_code[code] = row
    return {code for code, row in latest_by_code.items() if row["exit_date"] is None}
