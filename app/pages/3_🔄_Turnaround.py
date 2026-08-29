import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.data import load_latest_turnaround_predictions
from app.style import decision_badge, inject_base_css, regime_badge, render_developer_footer

st.set_page_config(page_title="MyStocks — Turnaround", page_icon="🔄", layout="wide")
inject_base_css()
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🔄 Turnaround Screener")
st.caption(
    "Saham yang SEDANG di regime bearish/bottoming, diberi skor probabilitas akan berhasil "
    "berbalik ke early_reversal/bullish dan bertahan di sana ≥20 hari perdagangan, dalam 6 bulan ke depan."
)
st.info(
    "**Beda karakter dari Screener Swing (Home)**: model ini punya base rate historis yang sudah tinggi "
    "(~82%), jadi nilainya bukan \"menemukan permata langka\" seperti sinyal BUY swing, melainkan lebih ke "
    "\"menyisihkan yang kemungkinan besar TIDAK akan berhasil\" (~15% dari kandidat). Precision di tingkat "
    "**POTENSIAL** terukur ~92% lewat walk-forward validation -- lihat halaman Info Model untuk detail.",
    icon="ℹ️",
)

df = load_latest_turnaround_predictions()
if df.empty:
    st.warning(
        "Belum ada prediksi turnaround. Jalankan `engine.predict_turnaround` terlebih dahulu "
        "(hanya menyekor ticker yang SAAT INI bearish/bottoming, jadi wajar kalau jumlahnya kecil)."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total kandidat (bearish/bottoming saat ini)", len(df))
c2.metric("POTENSIAL", int((df["decision"] == "POTENSIAL").sum()))
c3.metric("BELUM", int((df["decision"] == "BELUM").sum()))

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🔎 Filter")
    decision_filter = st.multiselect("Keputusan", ["POTENSIAL", "BELUM"], default=["POTENSIAL", "BELUM"])
    search = st.text_input("Cari kode/nama saham", placeholder="mis. ANTM atau bank")

filtered = df[df["decision"].isin(decision_filter)]
if search:
    q = search.strip().lower()
    filtered = filtered[
        filtered["stock_code"].str.lower().str.contains(q)
        | filtered["name"].fillna("").str.lower().str.contains(q)
    ]

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan filter saat ini.")
    st.stop()

st.subheader(f"📋 {len(filtered)} Kandidat — urut berdasarkan probabilitas")
st.caption("Klik header kolom untuk sortir ulang. Klik satu baris untuk buka halaman detail saham itu.")

table_df = filtered[["stock_code", "name", "sector", "industry", "decision", "probability", "entry_price", "regime"]].reset_index(drop=True)
table_df["probability"] = table_df["probability"].astype(float) * 100

event = st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_config={
        "stock_code": st.column_config.TextColumn("Kode"),
        "name": st.column_config.TextColumn("Nama"),
        "sector": st.column_config.TextColumn("Sektor"),
        "industry": st.column_config.TextColumn("Sub-sektor"),
        "decision": st.column_config.TextColumn("Status"),
        "probability": st.column_config.ProgressColumn("Probabilitas", format="%.1f%%", min_value=0.0, max_value=100.0),
        "entry_price": st.column_config.NumberColumn("Harga Saat Ini", format="%.0f"),
        "regime": st.column_config.TextColumn("Regime"),
    },
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    picked_code = table_df.iloc[selected_rows[0]]["stock_code"]
    st.session_state["selected_ticker"] = picked_code
    st.switch_page("pages/1_📈_Detail_Saham.py")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

st.subheader("🏆 Top Kandidat")
top = filtered.head(9)
CARDS_PER_ROW = 3
rows = [top.iloc[i:i + CARDS_PER_ROW] for i in range(0, len(top), CARDS_PER_ROW)]
for row_chunk in rows:
    cols = st.columns(CARDS_PER_ROW)
    for col, (_, r) in zip(cols, row_chunk.iterrows()):
        with col:
            # See Home.py's identical guard: NULL from SQL is float NaN, not
            # None, and NaN is truthy in Python -- `r["name"] or ...` would
            # silently print "nan" instead of falling back.
            name = r["stock_code"] if pd.isna(r["name"]) else r["name"]
            prob_pct = float(r["probability"]) * 100
            st.markdown(
                f"""
                <div class="mystocks-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div class="mystocks-ticker">{r['stock_code']}</div>
                            <div class="mystocks-muted">{name}</div>
                        </div>
                        {decision_badge(r['decision'])}
                    </div>
                    <div style="margin-top:0.7rem;">{regime_badge(r['regime'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(prob_pct / 100, 0.0), 1.0), text=f"Probabilitas: {prob_pct:.1f}%")
            sub1, sub2 = st.columns(2)
            sub1.markdown(f"<span class='mystocks-muted'>Harga Saat Ini</span><br>{float(r['entry_price']):,.0f}", unsafe_allow_html=True)
            industry_display = "-" if pd.isna(r["industry"]) else r["industry"]
            sub2.markdown(f"<span class='mystocks-muted'>Sub-sektor</span><br>{industry_display}", unsafe_allow_html=True)
            if st.button("Lihat Detail →", key=f"detail_{r['stock_code']}", width="stretch"):
                st.session_state["selected_ticker"] = r["stock_code"]
                st.switch_page("pages/1_📈_Detail_Saham.py")
            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
