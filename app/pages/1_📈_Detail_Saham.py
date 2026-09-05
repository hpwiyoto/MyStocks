import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
from plotly.subplots import make_subplots

from app.data import load_foreign_flow, load_foreign_flow_history, load_latest_feature_row, load_latest_fundamental, load_latest_predictions, load_live_prices, load_news, load_price_history, load_stock_list
from app.style import ACCENT, COLOR_AVOID, COLOR_BUY, decision_badge, inject_base_css, regime_badge, render_developer_footer, safe_ratio

def _notna(value):
    """`feat`/`fund` here are dicts built from a pandas row via .to_dict()
    -- a SQL NULL comes back as float NaN, not None, and `x is not None`
    doesn't catch NaN (it's a distinct float value). Used everywhere this
    page decides "is this value present" before formatting/displaying it,
    so a NULL field falls through to its "-" placeholder instead of
    silently rendering the literal string "nan"."""
    return value is not None and value == value


st.set_page_config(page_title="MyStocks — Detail Saham", page_icon="📈", layout="wide")
inject_base_css()
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

stocks_df = load_stock_list()
if stocks_df.empty:
    st.warning("Belum ada saham yang di-ingest. Jalankan pipeline terlebih dahulu dari halaman utama.")
    st.stop()

predictions = load_latest_predictions()

# Pemilihan bebas: dari seluruh saham yang pernah di-ingest, bukan cuma yang
# sudah punya prediksi hari ini -- tapi tetap kasih tahu mana yang belum.
scored_codes = set(predictions["stock_code"]) if not predictions.empty else set()
option_labels = {
    row["code"]: f"{row['code']} — {row['name']}" + ("" if row["code"] in scored_codes else " (belum ada prediksi)")
    for _, row in stocks_df.iterrows()
}
codes = stocks_df["code"].tolist()

# Root cause of the "kadang tidak langsung merespon" (sometimes doesn't
# immediately reflect the typed ticker) complaint, confirmed via Playwright:
# this selectbox had no explicit `key=`, so its displayed value was
# re-derived every rerun from `index=`, computed from
# session_state["selected_ticker"] -- which this same page ALSO wrote back
# into at the bottom. That write only lands AFTER the widget call, so on
# the very next rerun (the one Streamlit triggers immediately from the
# user's own pick) the value being read back was still the value from
# BEFORE that pick -- forcibly reverting the widget to the PREVIOUS ticker
# and clobbering whatever the user had just chosen.
#
# Correct fix: "selected_ticker" is only ever WRITTEN by *other* pages
# (Swing/Turnaround/Home's "Lihat Detail" buttons) right before
# switch_page() here -- a one-shot navigation signal, not a live mirror of
# this page's own state. Consume it exactly once via pop() to seed the
# widget's OWN key, then never touch either session_state entry again on
# this page -- Streamlit's own keyed-widget machinery is the sole source
# of truth for every subsequent interaction, so there is nothing left for
# this page to race against.
WIDGET_KEY = "detail_saham_ticker_select"
incoming = st.session_state.pop("selected_ticker", None)
if incoming in codes:
    st.session_state[WIDGET_KEY] = incoming
elif WIDGET_KEY not in st.session_state or st.session_state[WIDGET_KEY] not in codes:
    st.session_state[WIDGET_KEY] = codes[0]

selected = st.selectbox(
    "Pilih saham (bebas dari seluruh saham yang sudah di-ingest)",
    codes,
    format_func=lambda c: option_labels.get(c, c),
    key=WIDGET_KEY,
)

pred_match = predictions[predictions["stock_code"] == selected] if not predictions.empty else predictions
row = pred_match.iloc[0] if len(pred_match) else None
feat = load_latest_feature_row(selected)
fund = load_latest_fundamental(selected)

with st.spinner("Memuat data harga..."):
    price_df = load_price_history(selected, days=260)

# On-demand foreign-flow fetch+persist for whichever ticker is being
# viewed right now (see app/data.py's load_foreign_flow) -- display-only
# complement, deliberately NOT a model input (scripts/test_foreign_flow_feature.py
# found it doesn't help Swing's predictions). No-ops quietly if RAPIDAPI_KEY
# isn't configured. Cached FOREIGN_FLOW_TTL (6h), so this is cheap on repeat
# views of the same ticker in one sitting.
load_foreign_flow(selected)
foreign_flow_df = load_foreign_flow_history(selected, days=260)

stock_name = stocks_df.loc[stocks_df["code"] == selected, "name"].iloc[0] if selected in stocks_df["code"].values else ""

# Current price shown unconditionally, unlike the "Entry" metric below which
# only appears when this ticker has a swing prediction row -- a
# turnaround-only or unscored ticker previously showed no price at all here.
# Live quote reuses the same cheap per-ticker yfinance fast_info fetch as
# Swing's overlay (~0.3s, single ticker here); falls back to the last known
# close from price_df if the live fetch fails or the market's closed with no
# cached quote.
live_price = load_live_prices((selected,)).get(selected)
last_close = float(price_df["close"].iloc[-1]) if not price_df.empty else None
current_price = live_price if live_price is not None else last_close

# --- Header ---
h1, h2, h3 = st.columns([2.2, 1, 1])
with h1:
    badges = f"{decision_badge(row['decision'])} &nbsp; {regime_badge(row['regime'])}" if row is not None else ""
    st.markdown(
        f"""
        <div class="mystocks-ticker" style="font-size:2rem;">{selected} <span class="mystocks-muted" style="font-size:1.1rem;">{stock_name}</span></div>
        <div style="margin-top:0.4rem;">{badges}</div>
        """,
        unsafe_allow_html=True,
    )
with h2:
    st.markdown("<div class='mystocks-muted'>Harga Saat Ini</div>", unsafe_allow_html=True)
    if current_price is not None:
        st.markdown(f"<div class='mystocks-metric-value' style='font-size:2.2rem;'>{current_price:,.0f}</div>", unsafe_allow_html=True)
        st.caption("💹 Live" if live_price is not None else "Penutupan terakhir")
    else:
        st.markdown("<div class='mystocks-metric-value' style='font-size:2.2rem;'>-</div>", unsafe_allow_html=True)
with h3:
    if row is not None:
        st.markdown("<div class='mystocks-muted'>Probabilitas naik ≥5% sebelum SL -2.5% (10 hari)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='mystocks-metric-value' style='font-size:2.2rem;'>{float(row['probability'])*100:.1f}%</div>", unsafe_allow_html=True)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Entry / SL / TP ---
if row is not None:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Entry", f"{float(row['entry_price']):,.0f}")
    e2.metric("Stop Loss", f"{float(row['stop_loss_price']):,.0f}", delta=f"-{(1 - float(row['stop_loss_price'])/float(row['entry_price']))*100:.2f}%", delta_color="inverse")
    e3.metric("Take Profit", f"{float(row['take_profit_price']):,.0f}", delta=f"+{(float(row['take_profit_price'])/float(row['entry_price']) - 1)*100:.2f}%")
    e4.metric("Risk : Reward", f"1 : {float(row['risk_reward_ratio']):.1f}")
else:
    st.info(
        "Saham ini belum punya prediksi terbaru (belum di-scoring model, atau histori harganya masih terlalu pendek). "
        "Grafik & data fundamental di bawah tetap tersedia."
    )

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Candlestick chart ---
if price_df.empty:
    st.info("Belum ada data harga untuk saham ini.")
else:
    price_df = price_df.copy()
    price_df["ema5"] = price_df["close"].ewm(span=5, adjust=False).mean()
    price_df["ema9"] = price_df["close"].ewm(span=9, adjust=False).mean()
    price_df["bb_mid"] = price_df["close"].rolling(20).mean()
    bb_std = price_df["close"].rolling(20).std()
    price_df["bb_upper"] = price_df["bb_mid"] + 2 * bb_std
    price_df["bb_lower"] = price_df["bb_mid"] - 2 * bb_std
    price_df["sma50"] = price_df["close"].rolling(50).mean()
    price_df["sma200"] = price_df["close"].rolling(200).mean()
    # Naming note: this is the exact same computation (rolling(20).mean(),
    # a simple moving average) as sma50/sma200 above and sma_20/50/200 in
    # features/technical.py -- named "sma" (not "ma") for consistency with
    # every other moving average in this codebase; it was previously called
    # volume_ma20 with no functional difference, just an inconsistent label.
    price_df["volume_sma20"] = price_df["volume"].rolling(20).mean()
    # Same computation (ta library, window=20) the model itself uses for
    # cmf_20 -- see features/technical.py's compute_money_flow -- so this
    # panel matches exactly what the model sees, not a lookalike recomputed
    # differently.
    price_df["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(
        price_df["high"], price_df["low"], price_df["close"], price_df["volume"], window=20
    ).chaikin_money_flow()
    # Foreign flow isn't derivable from OHLCV -- merge in whatever's stored
    # in feature_daily.net_foreign_flow (kept current by load_foreign_flow's
    # on-demand fetch earlier on this page). Left join: a date with no
    # foreign-flow value (RAPIDAPI_KEY unset, or just not backfilled yet)
    # stays NaN, which Plotly simply skips/gaps rather than erroring on.
    price_df = price_df.merge(foreign_flow_df, on="date", how="left")

    INDICATOR_OPTIONS = ["EMA5", "EMA9", "SMA20", "SMA50", "SMA200", "Bollinger Band(20)"]
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        chart_type = st.radio("Tipe candle", ["Normal", "Heikin-Ashi"], horizontal=True, key="chart_type")
    with ctrl2:
        selected_indicators = st.multiselect(
            "Indikator ditampilkan", INDICATOR_OPTIONS, default=INDICATOR_OPTIONS, key="chart_indicators",
        )

    if chart_type == "Heikin-Ashi":
        ha_close = (price_df["open"] + price_df["high"] + price_df["low"] + price_df["close"]) / 4
        ha_open = ha_close.copy()
        ha_open.iloc[0] = (price_df["open"].iloc[0] + price_df["close"].iloc[0]) / 2
        for i in range(1, len(price_df)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
        ha_high = pd.concat([price_df["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([price_df["low"], ha_open, ha_close], axis=1).min(axis=1)
        plot_open, plot_high, plot_low, plot_close = ha_open, ha_high, ha_low, ha_close
        candle_name = "Harga (Heikin-Ashi)"
    else:
        plot_open, plot_high, plot_low, plot_close = price_df["open"], price_df["high"], price_df["low"], price_df["close"]
        candle_name = "Harga"

    # Row 3 (CMF) and row 4 (Foreign Flow) are always-on context panels, same
    # treatment as Volume (row 2) already gets -- not folded into the
    # INDICATOR_OPTIONS multiselect, since both are meant to be a permanent
    # complement to the price action rather than an optional overlay.
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.15, 0.20, 0.25], vertical_spacing=0.03,
        subplot_titles=(None, None, "CMF(20)", "Foreign Flow (Net Buy/Sell)"),
    )

    # Bollinger Band(20) shaded region -- drawn first so price/MA lines render on top
    if "Bollinger Band(20)" in selected_indicators:
        fig.add_trace(
            go.Scatter(x=price_df["date"], y=price_df["bb_upper"], name="BB Upper", line=dict(color="rgba(139,92,246,0.35)", width=1), showlegend=False),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=price_df["date"], y=price_df["bb_lower"], name="Bollinger Band(20)", line=dict(color="rgba(139,92,246,0.35)", width=1),
                fill="tonexty", fillcolor="rgba(139,92,246,0.08)",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=price_df["date"], open=plot_open, high=plot_high,
            low=plot_low, close=plot_close, name=candle_name,
            increasing_line_color=COLOR_BUY, decreasing_line_color=COLOR_AVOID,
        ),
        row=1, col=1,
    )
    for col, color, label in [
        ("ema5", "#FFFFFF", "EMA5"),
        ("ema9", "#EC4899", "EMA9"),
        ("bb_mid", "#F59E0B", "SMA20"),
        ("sma50", ACCENT, "SMA50"),
        ("sma200", "#8B5CF6", "SMA200"),
    ]:
        if label in selected_indicators:
            fig.add_trace(
                go.Scatter(x=price_df["date"], y=price_df[col], name=label, line=dict(color=color, width=1.3)),
                row=1, col=1,
            )
    volume_colors = [COLOR_BUY if c >= o else COLOR_AVOID for o, c in zip(price_df["open"], price_df["close"])]
    fig.add_trace(
        go.Bar(x=price_df["date"], y=price_df["volume"], name="Volume", marker_color=volume_colors, opacity=0.6),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=price_df["date"], y=price_df["volume_sma20"], name="SMA20 Volume", line=dict(color="#F59E0B", width=1.3)),
        row=2, col=1,
    )

    # CMF(20) -- oscillates roughly [-1, 1] around a zero line; zero-line
    # reference makes the buying/selling-pressure sign readable at a glance.
    fig.add_trace(
        go.Scatter(
            x=price_df["date"], y=price_df["cmf_20"], name="CMF(20)",
            line=dict(color="#22D3EE", width=1.3), showlegend=False,
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"), row=3, col=1)

    # Foreign Flow -- net foreign buy(+)/sell(-) in Rupiah, same up/down
    # bar-color convention as the Volume panel above. price_df only carries
    # a value on dates load_foreign_flow_history actually has (left-joined
    # above), so this naturally gaps rather than errors on missing days.
    if foreign_flow_df.empty:
        fig.add_annotation(
            text="Data foreign flow belum tersedia untuk emiten ini",
            xref="x domain", yref="y domain", x=0.5, y=0.5, row=4, col=1,
            showarrow=False, font=dict(color="rgba(255,255,255,0.45)", size=11),
        )
    else:
        ff_colors = [COLOR_BUY if v >= 0 else COLOR_AVOID for v in price_df["net_foreign_flow"].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=price_df["date"], y=price_df["net_foreign_flow"], name="Foreign Flow",
                marker_color=ff_colors, opacity=0.75, showlegend=False,
            ),
            row=4, col=1,
        )
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"), row=4, col=1)

    fig.update_layout(
        height=820,
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, width="stretch")
    if foreign_flow_df.empty:
        st.caption(
            "Foreign flow: data belum tersedia (RapidAPI key belum diset, atau "
            "belum pernah diambil untuk emiten ini). Data akan otomatis diambil "
            "saat halaman ini dibuka jika key sudah dikonfigurasi."
        )

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Panels: pattern similarity + fundamental ---
p1, p2 = st.columns(2)

with p1:
    st.subheader("🔁 Historical Pattern Similarity")
    if feat and _notna(feat.get("similarity_score")):
        st.metric("Similarity score (pola paling mirip)", f"{float(feat['similarity_score']):.2f}")
        pattern_count = feat.get("similar_pattern_count")
        st.metric("Jumlah pola historis serupa", int(pattern_count) if _notna(pattern_count) else 0)
        win_rate = feat.get("historical_win_rate")
        if _notna(win_rate):
            st.metric("Win rate pola serupa (10 hari ke depan)", f"{float(win_rate)*100:.1f}%")
        else:
            st.caption("Belum ada pola historis yang cukup mirip untuk dihitung win rate-nya.")
    else:
        st.caption("Data pattern similarity belum tersedia untuk saham ini.")

with p2:
    st.subheader("📊 Fundamental & Analis")
    if fund:
        f1, f2 = st.columns(2)
        f1.metric("P/E (trailing)", safe_ratio(fund.get("trailing_pe"), "{:.1f}"))
        f1.metric("P/B", safe_ratio(fund.get("price_to_book"), "{:.2f}"))
        f1.metric("Dividend Yield", f"{fund['dividend_yield']:.2f}%" if _notna(fund.get("dividend_yield")) else "-")
        f2.metric("ROE", f"{fund['return_on_equity']*100:.1f}%" if _notna(fund.get("return_on_equity")) else "-")
        f2.metric("Relative Strength vs IHSG (20d)", f"{fund['relative_strength_20d_pct']:.1f}%" if _notna(fund.get("relative_strength_20d_pct")) else "-")
        f2.metric("Analyst Upside", f"{fund['analyst_upside_pct']:.1f}%" if _notna(fund.get("analyst_upside_pct")) else "-")
        if "N/A*" in (safe_ratio(fund.get("trailing_pe"), "{:.1f}"), safe_ratio(fund.get("price_to_book"), "{:.2f}")):
            st.caption("*Data dari yfinance tidak wajar untuk rasio ini (terkonfirmasi: book value/EPS mendekati nol pada sumbernya), disembunyikan daripada menampilkan angka menyesatkan.")
    else:
        st.caption("Data fundamental belum tersedia untuk saham ini.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Sektor & Industri: konteks peer group, plus siapa lagi di industri yang
# sama sedang bagus sekarang (sector/industry sendiri sudah jadi bagian
# model lewat sector_relative_strength_20d_pct -- lihat features/build_
# features.py -- panel ini cuma menampilkannya secara visual). ---
st.subheader("🏭 Sektor & Industri")
stock_row = stocks_df[stocks_df["code"] == selected]
# A SQL NULL comes back from pandas as float NaN, not None -- and NaN is
# truthy in Python (bool(float('nan')) is True), so `sector or "-"` below
# would silently print the literal string "nan" instead of falling back.
# Normalize to real None right away rather than relying on truthiness.
sector = stock_row["sector"].iloc[0] if not stock_row.empty else None
industry = stock_row["industry"].iloc[0] if not stock_row.empty else None
sector = None if pd.isna(sector) else sector
industry = None if pd.isna(industry) else industry

if sector or industry:
    s1, s2, s3 = st.columns(3)
    s1.metric("Sektor", sector or "-")
    s2.metric("Sub-sektor (industri)", industry or "-")
    sect_rs = feat.get("sector_relative_strength_20d_pct") if feat else None
    s3.metric(
        "Relative Strength vs Sektor (20d)",
        f"{float(sect_rs):.1f}%" if _notna(sect_rs) else "-",
        help="Selisih return 20 hari saham ini vs rata-rata sektornya sendiri -- positif berarti mengungguli teman sesektor, bukan cuma pasar secara umum (lihat metrik 'Relative Strength vs IHSG' di panel Fundamental untuk pembanding pasar).",
    )

    if industry:
        peers = stocks_df[(stocks_df["industry"] == industry) & (stocks_df["code"] != selected)]
        if not peers.empty and not predictions.empty:
            peer_view = predictions[predictions["stock_code"].isin(peers["code"])][
                ["stock_code", "name", "decision", "probability", "regime"]
            ].sort_values("probability", ascending=False).head(8)
            if not peer_view.empty:
                st.caption(f"Saham lain di sub-sektor **{industry}** (diurutkan probabilitas):")
                peer_view = peer_view.copy()
                peer_view["probability"] = (peer_view["probability"].astype(float) * 100).round(1)
                st.dataframe(
                    peer_view, hide_index=True, width="stretch",
                    column_config={
                        "stock_code": st.column_config.TextColumn("Kode"),
                        "name": st.column_config.TextColumn("Nama"),
                        "decision": st.column_config.TextColumn("Keputusan"),
                        "probability": st.column_config.ProgressColumn("Probabilitas", format="%.1f%%", min_value=0.0, max_value=100.0),
                        "regime": st.column_config.TextColumn("Regime"),
                    },
                )
            else:
                st.caption(f"Belum ada prediksi terbaru untuk saham lain di sub-sektor {industry}.")
        else:
            st.caption(f"Tidak ada saham lain yang terdaftar di sub-sektor {industry}.")
else:
    st.caption("Data sektor/industri belum tersedia untuk saham ini.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Berita: panel tampilan saja, TIDAK masuk ke model (lihat features/news.py) ---
st.subheader("📰 Berita Terkait")
news_items = load_news(selected, stock_name)
if news_items:
    st.caption("Headline dari Google News, hanya untuk dibaca sendiri -- bukan bagian dari perhitungan probabilitas model.")
    for item in news_items:
        date_str = item["pub_date"].strftime("%d %b %Y") if item["pub_date"] else ""
        meta = " · ".join(p for p in (item["source"], date_str) if p)
        st.markdown(
            f"**[{item['title']}]({item['link']})**" + (f"  \n<span class='mystocks-muted'>{meta}</span>" if meta else ""),
            unsafe_allow_html=True,
        )
else:
    st.caption("Belum ada berita yang ditemukan untuk saham ini.")
