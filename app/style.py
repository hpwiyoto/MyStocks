"""Shared color palette, CSS, and small render helpers used across all pages.
Colors mirror .streamlit/config.toml so badges/charts match the app theme."""
import streamlit as st

BG = "#0B1120"
BG_CARD = "#151B2C"
BG_CARD_HOVER = "#1A2138"
BORDER = "#232B3E"
TEXT = "#E5E7EB"
TEXT_MUTED = "#8B95A7"
ACCENT = "#06B6D4"

COLOR_BUY = "#22C55E"
COLOR_WATCH = "#F59E0B"
COLOR_AVOID = "#EF4444"

REGIME_COLORS = {
    "bullish": "#22C55E",
    "early_reversal": "#06B6D4",
    "accumulation": "#3B82F6",
    "sideways": "#8B95A7",
    "bottoming": "#F59E0B",
    "bearish": "#EF4444",
    "overextended": "#F97316",
}

DECISION_COLORS = {"BUY": COLOR_BUY, "WATCH": COLOR_WATCH, "AVOID": COLOR_AVOID}


def inject_base_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}
        [data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {BORDER};
        }}
        .mystocks-card {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
            transition: border-color 0.15s ease;
        }}
        .mystocks-card:hover {{
            border-color: {ACCENT};
        }}
        .mystocks-badge {{
            display: inline-block;
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .mystocks-ticker {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {TEXT};
        }}
        .mystocks-muted {{
            color: {TEXT_MUTED};
            font-size: 0.85rem;
        }}
        .mystocks-metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {TEXT};
        }}
        .mystocks-divider {{
            border-top: 1px solid {BORDER};
            margin: 0.6rem 0;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str, color: str) -> str:
    return (
        f'<span class="mystocks-badge" style="background-color:{color}22; '
        f'color:{color}; border:1px solid {color}55;">{label}</span>'
    )


def decision_badge(decision: str) -> str:
    color = DECISION_COLORS.get(decision, TEXT_MUTED)
    return badge_html(decision or "-", color)


def regime_badge(regime: str) -> str:
    if not isinstance(regime, str):
        return badge_html("unknown", TEXT_MUTED)
    color = REGIME_COLORS.get(regime, TEXT_MUTED)
    return badge_html(regime.replace("_", " "), color)


def safe_ratio(value, fmt: str = "{:.2f}", max_abs: float = 100) -> str:
    """Format a valuation ratio (P/E, P/B), guarding against a confirmed
    upstream data quirk: yfinance's bookValue field is near-zero for some IDX
    tickers (e.g. ADRO: bookValue=0.17 vs price=2630), producing P/B ratios
    like 15470 that are real arithmetic but meaningless to show as-is. Values
    outside a generous sane bound are flagged instead of displayed raw."""
    if value is None:
        return "-"
    if abs(value) > max_abs:
        return "N/A*"
    return fmt.format(value)
