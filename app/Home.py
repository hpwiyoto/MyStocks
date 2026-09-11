import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.auth import require_login
from app.data import (
    load_data_freshness,
    load_ihsg_history,
    load_ihsg_trend,
    load_latest_predictions,
    load_screener_raw_panel,
    load_stock_list,
)
from app.style import data_freshness_note, inject_base_css, render_developer_footer, render_ihsg_chart, render_ihsg_context
from features.momentum_screener import compute_screener_panel

st.set_page_config(page_title="MyStocks — Home", page_icon="🏠", layout="wide")
inject_base_css()
require_login("Home")

st.title("🏠 MyStocks")
st.caption(
    "Screener saham IDX berbasis machine learning. Pilih mode di bawah sesuai gaya trading/investasi Anda, "
    "atau cari langsung satu saham tertentu."
)
render_ihsg_context(load_ihsg_trend())

IHSG_PERIOD_OPTIONS = {"1 Bulan": "1mo", "3 Bulan": "3mo", "6 Bulan": "6mo", "1 Tahun": "1y"}
_ihsg_period_label = st.radio(
    "Periode grafik IHSG", list(IHSG_PERIOD_OPTIONS.keys()), index=2, horizontal=True,
    label_visibility="collapsed", key="home_ihsg_period",
)
render_ihsg_chart(load_ihsg_history(IHSG_PERIOD_OPTIONS[_ihsg_period_label]))

data_freshness_note(load_data_freshness())

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Pencarian cepat: langsung ke Detail Saham, tidak perlu lewat Swing
# dulu kalau Anda sudah tahu kode sahamnya. ---
st.subheader("🔎 Cari Saham")
stocks_df = load_stock_list()
codes = stocks_df["code"].tolist() if not stocks_df.empty else []
quick_search = st.text_input(
    "Ketik kode saham", placeholder="mis. BBCA", label_visibility="collapsed",
)
if quick_search:
    match = quick_search.strip().upper()
    if match in codes:
        st.session_state["selected_ticker"] = match
        st.switch_page("pages/1_📈_Detail_Saham.py")
    else:
        st.caption(f"Kode `{match}` tidak ditemukan di daftar saham yang sudah di-ingest.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

st.subheader("📊 Pilih Mode Screening")

swing_df = load_latest_predictions()
# Momentum Screener is rule-based (not a trained model, see the page itself
# for the full backtest story) -- computed here too, cheaply, just for the
# "berapa sinyal tervalidasi hari ini" count below. Previously this whole
# section only showed Swing/Turnaround, silently leaving Momentum Screener
# and Rekomendasi Emitten -- two fully-built tools -- completely off the
# home page, as if they didn't exist.
momentum_raw = load_screener_raw_panel(lookback_days=60)
momentum_df = compute_screener_panel(momentum_raw) if not momentum_raw.empty else momentum_raw
validated_total = int(momentum_df["validated_signal"].sum()) if not momentum_df.empty else 0

# Cheap set-membership version of Rekomendasi Emitten's own agreement-count
# logic (see that page for the full merge/display) -- just enough here for
# a "sekian saham disepakati" headline count, not a full recomputation.
# Turnaround dropped from this consensus entirely (see app/pages/5's
# docstring/comments -- its top-2 real-world performance came in well
# below Swing's even on Turnaround's OWN proposed target, and the existing
# 6-month model is retired) -- Swing + Momentum only now, 2 alat, not 3.
swing_hit = set(swing_df.loc[swing_df["decision"].isin(["BUY", "WATCH"]), "stock_code"]) if not swing_df.empty else set()
momentum_hit = set(momentum_df.loc[momentum_df["validated_signal"], "stock_code"]) if not momentum_df.empty else set()
all_codes = stocks_df["code"].tolist() if not stocks_df.empty else []
agreement_counts = [(c in swing_hit) + (c in momentum_hit) for c in all_codes]
consensus_2of2 = sum(1 for a in agreement_counts if a == 2)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        <div class="mystocks-card">
            <div class="mystocks-ticker" style="font-size:1.3rem;">🎯 Swing (10 hari)</div>
            <div class="mystocks-muted" style="min-height:3.9em; line-height:1.3em;">Cari peluang naik ≥5% sebelum stop-loss -2.5% dalam 10 hari trading.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not swing_df.empty:
        s1, s2 = st.columns(2)
        s1.metric("BUY", int((swing_df["decision"] == "BUY").sum()))
        s2.metric("WATCH", int((swing_df["decision"] == "WATCH").sum()))
    else:
        st.caption("Belum ada data prediksi.")
    if st.button("Buka Swing Screener →", key="goto_swing", width="stretch"):
        st.switch_page("pages/2_🎯_Swing.py")

with c2:
    st.markdown(
        """
        <div class="mystocks-card">
            <div class="mystocks-ticker" style="font-size:1.3rem;">📡 Momentum Screener</div>
            <div class="mystocks-muted" style="min-height:3.9em; line-height:1.3em;">Filter RSI/MACD/volume/money flow -- aturan teknikal, bukan model ML.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not momentum_df.empty:
        m1, m2 = st.columns(2)
        m1.metric("✅ Tervalidasi", validated_total)
        m2.metric("Total dipantau", len(momentum_df))
    else:
        st.caption("Belum ada data harga/fitur.")
    if st.button("Buka Momentum Screener →", key="goto_momentum", width="stretch"):
        st.switch_page("pages/4_📡_Momentum_Screener.py")

st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)

with d1:
    st.markdown(
        """
        <div class="mystocks-card">
            <div class="mystocks-ticker" style="font-size:1.3rem;">🏆 Rekomendasi Emitten</div>
            <div class="mystocks-muted" style="min-height:3.9em; line-height:1.3em;">Saham yang disepakati kedua alat (Swing + Momentum) sekaligus.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if all_codes:
        st.metric("🌟 Keduanya Sepakat (2/2)", consensus_2of2)
    else:
        st.caption("Belum ada data.")
    if st.button("Buka Rekomendasi Emitten →", key="goto_rekomendasi", width="stretch"):
        st.switch_page("pages/5_🏆_Rekomendasi_Emitten.py")

with d2:
    st.markdown(
        """
        <div class="mystocks-card" style="opacity:0.6;">
            <div class="mystocks-ticker" style="font-size:1.3rem;">📈 Long-term Investment</div>
            <div class="mystocks-muted" style="min-height:3.9em; line-height:1.3em;">Cari perusahaan yang mengungguli IHSG dalam 12 bulan. Segera hadir.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Belum dibangun -- definisi target masih dalam perancangan.")

render_developer_footer()
