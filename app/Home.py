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
    search = st.text_input("Cari kode/nama saham", placeholder="mis. BBCA atau bank")
    decision_filter = st.multiselect("Keputusan", ["BUY", "WATCH", "AVOID"], default=["BUY", "WATCH", "AVOID"])
    min_prob = st.slider("Probabilitas minimum", 0.0, 1.0, 0.0, 0.05)
    price_filter = st.selectbox(
        "Harga saham",
        ["Semua", "Di bawah 50", "50 - 100", "100 - 1.000", "Di atas 1.000"],
    )
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
price = filtered["entry_price"].astype(float)
if price_filter == "Di bawah 50":
    filtered = filtered[price < 50]
elif price_filter == "50 - 100":
    filtered = filtered[(price >= 50) & (price < 100)]
elif price_filter == "100 - 1.000":
    filtered = filtered[(price >= 100) & (price < 1000)]
elif price_filter == "Di atas 1.000":
    filtered = filtered[price >= 1000]
if search:
    q = search.strip().lower()
    filtered = filtered[
        filtered["stock_code"].str.lower().str.contains(q)
        | filtered["name"].fillna("").str.lower().str.contains(q)
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

# --- Top peluang: highlight kartu untuk beberapa probabilitas tertinggi ---
TOP_N_CARDS = 9
top = filtered.head(TOP_N_CARDS)
st.subheader(f"🏆 Top {min(TOP_N_CARDS, len(top))} Peluang Tertinggi")

CARDS_PER_ROW = 3
rows = [top.iloc[i : i + CARDS_PER_ROW] for i in range(0, len(top), CARDS_PER_ROW)]
for row_chunk in rows:
    cols = st.columns(CARDS_PER_ROW)
    for col, (_, r) in zip(cols, row_chunk.iterrows()):
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

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Tabel ranking penuh: scalable untuk ratusan ticker (klik baris untuk detail) ---
st.subheader(f"📋 Semua Saham ({len(filtered)}) — urut berdasarkan probabilitas")
st.caption("Klik header kolom untuk sortir ulang. Klik satu baris untuk buka halaman detail saham itu.")

table_df = filtered[["stock_code", "name", "decision", "probability", "regime", "entry_price", "stop_loss_price", "take_profit_price"]].reset_index(drop=True)
table_df["probability"] = table_df["probability"].astype(float) * 100  # ProgressColumn format="%.1f%%" doesn't auto-scale from 0-1

event = st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_config={
        "stock_code": st.column_config.TextColumn("Kode"),
        "name": st.column_config.TextColumn("Nama"),
        "decision": st.column_config.TextColumn("Keputusan"),
        "probability": st.column_config.ProgressColumn("Probabilitas", format="%.1f%%", min_value=0.0, max_value=100.0),
        "regime": st.column_config.TextColumn("Regime"),
        "entry_price": st.column_config.NumberColumn("Entry", format="%.0f"),
        "stop_loss_price": st.column_config.NumberColumn("Stop Loss", format="%.0f"),
        "take_profit_price": st.column_config.NumberColumn("Take Profit", format="%.0f"),
    },
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    picked_code = table_df.iloc[selected_rows[0]]["stock_code"]
    st.session_state["selected_ticker"] = picked_code
    st.switch_page("pages/1_📈_Detail_Saham.py")
