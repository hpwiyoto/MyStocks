"""Shared color palette, CSS, and small render helpers used across all pages.
Colors mirror .streamlit/config.toml so badges/charts match the app theme."""
import datetime as dt
import textwrap

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

DECISION_COLORS = {
    "BUY": COLOR_BUY, "WATCH": COLOR_WATCH, "AVOID": COLOR_AVOID,
    "POTENSIAL": COLOR_BUY, "BELUM": COLOR_WATCH,  # turnaround screener's own decision tiers
}


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
            white-space: nowrap;
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


def render_ihsg_context(trend: dict | None):
    """Compact IHSG trend banner -- see app/data.py's load_ihsg_trend()
    docstring for the full reasoning (scripts/check_fold_drift.py found
    Swing's precision correlates with IHSG's direction; feeding IHSG's
    trend INTO the model made things worse, per
    scripts/test_ihsg_regime_feature.py -- this surfaces the same context
    to the human instead). No-ops quietly if trend is None/incomplete
    (fetch failed) rather than showing a broken banner.
    """
    if not trend or trend.get("ret_20d") is None:
        return
    ret_20d = trend["ret_20d"]
    ret_50d = trend.get("ret_50d")
    if ret_20d <= -5:
        icon, color = "📉", COLOR_AVOID
        note = "historis performa model Swing lebih lemah saat IHSG turun seperti ini -- lihat halaman Info Model."
    elif ret_20d < 0:
        icon, color = "📉", COLOR_WATCH
        note = "IHSG sedang melemah -- lihat halaman Info Model untuk konteks performa model saat ini."
    else:
        icon, color = "📈", COLOR_BUY
        note = "IHSG sedang menguat."
    ret_50d_txt = f" &middot; {ret_50d:+.1f}% (50 hari)" if ret_50d is not None else ""
    html = textwrap.dedent(f"""
        <div style="background-color:{color}14; border:1px solid {color}44; border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.8rem;">
            <span style="color:{color}; font-weight:600;">{icon} IHSG {ret_20d:+.1f}% (20 hari){ret_50d_txt}</span>
            <span class="mystocks-muted"> -- {note}</span>
        </div>
        """)
    st.markdown(html, unsafe_allow_html=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """'#22C55E' + 0.1 -> 'rgba(34,197,94,0.1)'. Plotly's `fillcolor` only
    accepts 6-digit hex/rgb/rgba/hsl/named colors -- NOT the 8-digit
    hex-with-alpha shorthand ('#22C55E1A') the raw-HTML/CSS badges
    elsewhere in this module can get away with (browsers accept that
    syntax; plotly.py's own color validator rejects it outright, confirmed
    live via a ValueError crash on this exact chart)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_ihsg_chart(hist, height: int = 130) -> None:
    """Small line chart (sparkline-ish, not a full-size chart) of IHSG's
    own close over the lookback in `hist` (a date/close DataFrame from
    app.data.load_ihsg_history()), shown right under render_ihsg_context's
    text banner on Home. Plain st.plotly_chart -- NOT the custom-JS
    cross-panel crosshair setup Detail Saham's 6-panel chart uses (see
    detail-saham-chart-architecture notes): that machinery exists purely
    to keep several panels' crosshairs in sync, and a single line here has
    nothing to sync with, so plotly's own built-in hover is simplest and
    sufficient. theme=None so the dark palette below applies as-is --
    Streamlit's default 'streamlit' plotly theme fights the app's dark
    background otherwise.

    Y-axis range is pinned tight to the data's own min/max (+ a small
    padding) instead of trusting plotly's autorange -- IHSG only moves a
    few percent over most of these lookbacks, and on a short 130px-tall
    chart, autorange's default padding was flattening genuine moves into
    what looked like a barely-visible line (real user feedback: "naik
    turunnya tidak terlihat signifikan"). Pinning the range to the actual
    data spread makes the same move fill the whole chart height instead.

    Shows a caption instead of silently rendering nothing if hist is
    None/empty (fetch failed) -- a prior version no-op'd here, which read
    as "the chart is just broken" with zero indication why (real user
    report: picking a period showed nothing, no error, no explanation).
    """
    import plotly.graph_objects as go

    if hist is None or hist.empty:
        st.caption("📉 Grafik IHSG tidak tersedia (gagal ambil data dari sumbernya) -- coba lagi beberapa saat lagi.")
        return
    first, last = float(hist["close"].iloc[0]), float(hist["close"].iloc[-1])
    color = COLOR_BUY if last >= first else COLOR_AVOID
    y_min, y_max = float(hist["close"].min()), float(hist["close"].max())
    pad = (y_max - y_min) * 0.12 or y_max * 0.005  # flat-line guard (y_max==y_min)
    span_days = (hist["date"].max() - hist["date"].min()).days
    # Fewer, cleanly-formatted ticks instead of plotly's own date auto-
    # ticking, which (confirmed real: user feedback, dates along the
    # bottom looked cramped/off) was free to pack in far more labels than
    # a chart this short has vertical room for. "1 Bulan"/"3 Bulan" show
    # day+month (the year never changes within the window, so it'd be
    # redundant); "1 Tahun" switches to month+year since it crosses a
    # year boundary and a bare day+month would be ambiguous about which
    # occurrence.
    x_tickformat = "%d %b" if span_days <= 200 else "%b '%y"
    fig = go.Figure(go.Scatter(
        x=hist["date"], y=hist["close"], mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hovertemplate="%{x|%d %b %Y}<br>IHSG %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=4, b=22),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=10),
        xaxis=dict(
            showgrid=False, color=TEXT_MUTED, fixedrange=True,
            tickformat=x_tickformat, nticks=5, tickangle=0,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=BORDER, color=TEXT_MUTED, tickformat=",.0f",
            nticks=3, fixedrange=True, range=[y_min - pad, y_max + pad],
        ),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def render_developer_footer():
    """Sidebar footer shown on every page -- developer contact info."""
    st.sidebar.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size:0.8rem; line-height:1.6;">
            <div class="mystocks-muted">Dikembangkan oleh</div>
            <div style="font-weight:600;">Heru Purbo Wiyoto</div>
            <div class="mystocks-muted">📱 0811 1299 599</div>
            <div class="mystocks-muted">✉️ heru.purbowiyoto@gmail.com</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def data_freshness_note(freshness_date, *, sidebar: bool = False) -> None:
    """One-line 'data per <tanggal>' status, shown on every page so a
    silently-stale scheduler (laptop asleep past the 16:30 WIB slot -- a
    real incident, see scripts/scheduler_loop.py) is visible everywhere,
    not just on the Momentum Screener where this check first lived.

    `freshness_date` is app.data.load_data_freshness()'s return -- the most
    recent date in feature_daily -- or None. Passed in by the caller rather
    than imported here to keep app.style free of an app.data dependency.
    `sidebar=True` renders into st.sidebar (for pages whose main column
    opens with a big custom header)."""
    target = st.sidebar if sidebar else st
    if freshness_date is None:
        target.caption("⚠️ Status kesegaran data tidak diketahui (feature_daily kosong).")
        return
    today = dt.date.today()
    # Weekday-only gap: IDX doesn't trade Sat/Sun, so a plain calendar diff
    # would falsely flag every Monday as stale. Doesn't know IDX public
    # holidays specifically -- a much rarer false positive than weekends.
    age = sum(
        1 for i in range(1, (today - freshness_date).days + 1)
        if (freshness_date + dt.timedelta(days=i)).weekday() < 5
    )
    label = freshness_date.strftime("%d %b %Y")
    if age >= 2:
        target.warning(
            f"⚠️ Data harga & indikator terakhir per **{label}** ({age} hari bursa lalu) -- "
            "scheduler kemungkinan sempat tidak jalan (mis. laptop mati/tidur). Kalau update "
            "otomatis sedang jalan, refresh halaman ini beberapa menit lagi."
        )
    elif age == 1:
        target.caption(f"🕒 Data per {label} (kemarin) -- normal untuk pagi hari sebelum jadwal update sore ini.")
    else:
        target.caption(f"✅ Data per {label} (hari ini).")


# (label, threshold in Rupiah). First entry (None) is the default -- a
# hard liquidity cut was backtested (scripts/test_liquidity_filter.py) and
# does NOT improve any screener's top-2 hit rate (it noticeably hurt
# Swing's: 69% -> ~50%), so this filter is strictly opt-in, never on by
# default.
LIQUIDITY_FILTER_OPTIONS = [
    ("Semua (tanpa filter likuiditas)", None),
    ("≥ Rp 500 juta/hari", 0.5e9),
    ("≥ Rp 1 miliar/hari", 1e9),
    ("≥ Rp 2 miliar/hari", 2e9),
    ("≥ Rp 5 miliar/hari", 5e9),
]


def format_traded_value(value) -> str:
    """Rupiah traded-value, compact ('Rp 4,4 M' / 'Rp 710 jt' / 'Rp 12 rb').
    NaN/None -> '-' (same SQL-NULL-is-float-NaN guard as safe_ratio)."""
    if value is None or value != value:
        return "-"
    v = float(value)
    if v >= 1e9:
        return f"Rp {v / 1e9:,.1f} M".replace(",", ".")
    if v >= 1e6:
        return f"Rp {v / 1e6:,.0f} jt".replace(",", ".")
    if v >= 1e3:
        return f"Rp {v / 1e3:,.0f} rb".replace(",", ".")
    return f"Rp {v:,.0f}".replace(",", ".")


def liquidity_sidebar_filter(key: str) -> float | None:
    """Sidebar selectbox for a minimum average-daily-traded-value gate,
    shared by all four screener pages. Returns the threshold in Rupiah, or
    None for 'no filter' (the default -- see LIQUIDITY_FILTER_OPTIONS).
    `key` must be unique per page (Streamlit widget keys are global)."""
    labels = [o[0] for o in LIQUIDITY_FILTER_OPTIONS]
    choice = st.sidebar.selectbox(
        "Likuiditas minimum (rata-rata transaksi 60 hari)",
        labels, index=0, key=key,
        help=(
            "Saring ke saham yang cukup ramai ditransaksikan supaya sinyalnya lebih "
            "mencerminkan harga yang benar-benar bisa dieksekusi (slippage/spread kecil). "
            "Default: tanpa filter -- backtest menunjukkan filter ini TIDAK menaikkan "
            "akurasi (malah menurunkan untuk Swing), jadi murni opsional untuk yang "
            "hanya mau nama likuid."
        ),
    )
    return dict(LIQUIDITY_FILTER_OPTIONS)[choice]


def safe_ratio(value, fmt: str = "{:.2f}", max_abs: float = 100) -> str:
    """Format a valuation ratio (P/E, P/B), guarding against a confirmed
    upstream data quirk: yfinance's bookValue field is near-zero for some IDX
    tickers (e.g. ADRO: bookValue=0.17 vs price=2630), producing P/B ratios
    like 15470 that are real arithmetic but meaningless to show as-is. Values
    outside a generous sane bound are flagged instead of displayed raw."""
    # SQL NULL comes back from pandas as float NaN, not None -- and NaN
    # fails every ordinary comparison (`abs(nan) > max_abs` is False, not
    # an error), so without this check a NULL ratio would silently render
    # as the literal string "nan" instead of falling back to "-".
    # `value != value` is the dependency-free NaN test (NaN is the only
    # float where that's ever True).
    if value is None or value != value:
        return "-"
    if abs(value) > max_abs:
        return "N/A*"
    return fmt.format(value)
