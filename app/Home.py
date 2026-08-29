import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.data import load_latest_predictions, load_latest_turnaround_predictions, load_stock_list
from app.style import inject_base_css, render_developer_footer

st.set_page_config(page_title="MyStocks — Home", page_icon="🏠", layout="wide")
inject_base_css()

st.title("🏠 MyStocks")
st.caption(
    "Screener saham IDX berbasis machine learning. Pilih mode di bawah sesuai gaya trading/investasi Anda, "
    "atau cari langsung satu saham tertentu."
)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Pencarian cepat: langsung ke Detail Saham, tidak perlu lewat Swing/
# Turnaround dulu kalau Anda sudah tahu kode sahamnya. ---
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
turnaround_df = load_latest_turnaround_predictions()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="mystocks-card">
            <div class="mystocks-ticker" style="font-size:1.3rem;">🎯 Swing (10 hari)</div>
            <div class="mystocks-muted">Cari peluang naik ≥5% sebelum stop-loss -2.5% dalam 10 hari trading.</div>
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
            <div class="mystocks-ticker" style="font-size:1.3rem;">🔄 Turnaround</div>
            <div class="mystocks-muted">Saham bearish/bottoming yang berpotensi berbalik arah dalam 6 bulan.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not turnaround_df.empty:
        t1, t2 = st.columns(2)
        t1.metric("POTENSIAL", int((turnaround_df["decision"] == "POTENSIAL").sum()))
        t2.metric("Total kandidat", len(turnaround_df))
    else:
        st.caption("Belum ada data prediksi.")
    if st.button("Buka Turnaround Screener →", key="goto_turnaround", width="stretch"):
        st.switch_page("pages/3_🔄_Turnaround.py")

with c3:
    st.markdown(
        """
        <div class="mystocks-card" style="opacity:0.6;">
            <div class="mystocks-ticker" style="font-size:1.3rem;">📈 Long-term Investment</div>
            <div class="mystocks-muted">Cari perusahaan yang mengungguli IHSG dalam 12 bulan. Segera hadir.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Belum dibangun -- definisi target masih dalam perancangan.")

render_developer_footer()
