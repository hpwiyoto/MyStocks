import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.data import load_latest_predictions
from app.style import decision_badge, inject_base_css, regime_badge

st.set_page_config(page_title="MyStocks — Screener", page_icon="📈", layout="wide")
inject_base_css()

st.title("📈 MyStocks Screener")
st.caption("Prediksi harian saham IDX — probabilitas naik ≥5% sebelum stop-loss -2.5% dalam 10 hari trading.")

with st.sidebar:
    st.header("🔎 Filter")
    decision_filter = st.multiselect("Keputusan", ["BUY", "WATCH", "AVOID"], default=["BUY", "WATCH", "AVOID"])
    min_prob = st.slider("Probabilitas minimum", 0.0, 1.0, 0.0, 0.05)
    st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Memuat prediksi terbaru..."):
    df = load_latest_predictions()

if df.empty:
    st.warning(
        "Belum ada data prediksi. Jalankan `pipeline.ingest_price` → `features.build_features` "
        "→ `engine.predict` terlebih dahulu."
    )
    st.stop()

regime_options = sorted(df["regime"].dropna().unique().tolist())
with st.sidebar:
    regime_filter = st.multiselect("Regime", regime_options, default=regime_options)

filtered = df[
    df["decision"].isin(decision_filter)
    & df["probability"].astype(float).ge(min_prob)
    & (df["regime"].isin(regime_filter) | df["regime"].isna())
]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total ticker", len(df))
col2.metric("BUY", int((df["decision"] == "BUY").sum()))
col3.metric("WATCH", int((df["decision"] == "WATCH").sum()))
col4.metric("Ditampilkan", len(filtered))

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan filter saat ini.")
    st.stop()

CARDS_PER_ROW = 3
rows = [filtered.iloc[i : i + CARDS_PER_ROW] for i in range(0, len(filtered), CARDS_PER_ROW)]

for row in rows:
    cols = st.columns(CARDS_PER_ROW)
    for col, (_, r) in zip(cols, row.iterrows()):
        with col:
            name = r["name"] or r["stock_code"]
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
            st.progress(min(max(float(r["probability"]), 0.0), 1.0), text=f"Probabilitas: {prob_pct:.1f}%")
            e1, e2, e3 = st.columns(3)
            e1.markdown(f"<span class='mystocks-muted'>Entry</span><br>{float(r['entry_price']):,.0f}", unsafe_allow_html=True)
            e2.markdown(f"<span class='mystocks-muted'>Stop Loss</span><br>{float(r['stop_loss_price']):,.0f}", unsafe_allow_html=True)
            e3.markdown(f"<span class='mystocks-muted'>Take Profit</span><br>{float(r['take_profit_price']):,.0f}", unsafe_allow_html=True)
            if st.button("Lihat Detail →", key=f"detail_{r['stock_code']}", width="stretch"):
                st.session_state["selected_ticker"] = r["stock_code"]
                st.switch_page("pages/1_📈_Detail_Saham.py")
            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
