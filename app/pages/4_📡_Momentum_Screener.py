import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.data import load_data_freshness, load_latest_predictions, load_screener_raw_panel, load_stock_list
from app.style import (
    ACCENT,
    COLOR_AVOID,
    COLOR_BUY,
    TEXT_MUTED,
    badge_html,
    inject_base_css,
    regime_badge,
    render_developer_footer,
)
from features.momentum_screener import compute_screener_panel

st.set_page_config(page_title="MyStocks — Momentum Screener", page_icon="📡", layout="wide")
inject_base_css()
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("📡 Momentum Screener")
st.caption(
    "Filter manual berbasis RSI, status MACD, harga, volume, dan money flow (CMF) -- "
    "BUKAN skor model. Prioritas utama diberikan ke saham dengan divergence bullish "
    "(harga lower low, indikator higher low); probabilitas model Swing ditampilkan "
    "cuma sebagai referensi tambahan, bukan penentu urutan."
)
st.info(
    "**Beda dari Swing/Turnaround**: dua screener lain di aplikasi ini diurutkan oleh "
    "probabilitas model machine learning. Screener ini sebaliknya -- urutannya murni "
    "dari aturan teknikal yang Anda tentukan sendiri (RSI/MACD/volume/money flow), dan "
    "divergence selalu diprioritaskan di atas. Kolom probabilitas tetap ditampilkan "
    "supaya Anda bisa membandingkan, bukan supaya menggantikan penilaian teknikal ini.",
    icon="ℹ️",
)

# Harga & semua indikator (RSI/MACD/CMF) di halaman ini datang dari
# feature_daily, yang cuma seaktual scheduler harian (lihat
# scripts/scheduler_loop.py) -- kalau mesin yang menjalankan scheduler
# sempat mati/tidur, datanya bisa diam-diam basi tanpa tanda apa pun.
# Ditampilkan eksplisit di sini setelah kejadian nyata: 4 hari basi tanpa
# ada yang sadar sampai ditanyakan langsung.
_freshness = load_data_freshness()
if _freshness is not None:
    _age_days = (dt.date.today() - _freshness).days
    if _age_days >= 2:
        st.warning(
            f"⚠️ Data harga & indikator terakhir per **{_freshness.strftime('%d %b %Y')}** "
            f"({_age_days} hari lalu) -- scheduler kemungkinan sempat tidak jalan (mis. laptop "
            "mati/tidur). Sedang di-update otomatis di background; refresh halaman ini beberapa "
            "menit lagi untuk data terbaru.",
        )
    elif _age_days == 1:
        st.caption(f"🕒 Data per {_freshness.strftime('%d %b %Y')} (kemarin) -- normal untuk pagi hari sebelum jadwal update sore ini.")
    else:
        st.caption(f"✅ Data per {_freshness.strftime('%d %b %Y')} (hari ini).")

DIVERGENCE_LABELS = {0: "🔥 Ganda (RSI+MACD)", 1: "Tunggal", 2: "-"}
DIVERGENCE_COLORS = {0: "#A855F7", 1: ACCENT, 2: TEXT_MUTED}
MACD_STATUS_COLORS = {
    "Bullish Crossover": COLOR_BUY, "Bullish": COLOR_BUY,
    "Bearish Crossover": COLOR_AVOID, "Bearish": COLOR_AVOID,
    "Netral": TEXT_MUTED, "Tidak diketahui": TEXT_MUTED,
}
PRICE_FILTER_OPTIONS = ["Semua", "Di bawah 50", "50 - 100", "100 - 1.000", "Di atas 1.000"]


def divergence_detail(row) -> str:
    if row["divergence_tier"] == 2:
        return "-"
    parts = []
    if row["divergence_rsi"]:
        parts.append("RSI")
    if row["divergence_macd"]:
        parts.append("MACD")
    age = row["divergence_age_days"]
    age_txt = f", {int(age)}h lalu" if pd.notna(age) else ""
    return f"{'+'.join(parts)}{age_txt}"


def momentum_label(macd_hist, slope) -> str:
    if pd.isna(macd_hist) or pd.isna(slope):
        return "-"
    if macd_hist >= 0:
        return "Momentum menguat ↑" if slope > 0 else "Momentum melemah ↓"
    return "Tekanan jual melemah ↑" if slope > 0 else "Tekanan jual menguat ↓"


with st.spinner("Menghitung status RSI/MACD/divergence untuk seluruh saham..."):
    raw_panel = load_screener_raw_panel(lookback_days=60)

if raw_panel.empty:
    st.warning("Belum ada data harga/fitur yang cukup. Jalankan pipeline & features terlebih dahulu.")
    st.stop()

screener_df = compute_screener_panel(raw_panel)
stocks_df = load_stock_list()
predictions = load_latest_predictions()

df = screener_df.merge(stocks_df, left_on="stock_code", right_on="code", how="left")
df = df.merge(
    predictions[["stock_code", "probability"]] if not predictions.empty
    else pd.DataFrame(columns=["stock_code", "probability"]),
    on="stock_code", how="left",
)

with st.sidebar:
    st.header("🔎 Filter")
    search = st.text_input("Cari kode/nama saham", placeholder="mis. BBCA atau bank")

    rsi_range = st.slider("Rentang RSI", 0, 100, (20, 60))

    macd_options = sorted(df["macd_status"].dropna().unique().tolist())
    macd_default = [o for o in ["Bullish Crossover", "Bullish"] if o in macd_options] or macd_options
    macd_filter = st.multiselect("Status MACD", macd_options, default=macd_default)

    money_flow_filter = st.radio(
        "Money Flow (CMF 20 hari)", ["Semua", "Akumulasi (CMF > 0)", "Distribusi (CMF < 0)"],
        index=1,
    )

    volume_filter = st.checkbox("Hanya volume di atas rata-rata (RVOL ≥ 1)", value=False)

    price_filter = st.selectbox("Harga saham", PRICE_FILTER_OPTIONS, index=0)
    st.caption("⚠️ Saham di bawah Rp50 (gocap) tidak dikecualikan di sini seperti di Swing -- likuiditas & tick-size-nya perlu ekstra hati-hati.")

    divergence_only = st.checkbox("Hanya yang ada divergence bullish", value=False)

filtered = df[df["rsi_14"].between(rsi_range[0], rsi_range[1])]
if macd_filter:
    filtered = filtered[filtered["macd_status"].isin(macd_filter)]
if money_flow_filter == "Akumulasi (CMF > 0)":
    filtered = filtered[filtered["cmf_20"] > 0]
elif money_flow_filter == "Distribusi (CMF < 0)":
    filtered = filtered[filtered["cmf_20"] < 0]
if volume_filter:
    filtered = filtered[filtered["rvol_20"] >= 1]
price = filtered["close"].astype(float)
if price_filter == "Di bawah 50":
    filtered = filtered[price < 50]
elif price_filter == "50 - 100":
    filtered = filtered[(price >= 50) & (price < 100)]
elif price_filter == "100 - 1.000":
    filtered = filtered[(price >= 100) & (price < 1000)]
elif price_filter == "Di atas 1.000":
    filtered = filtered[price >= 1000]
if divergence_only:
    filtered = filtered[filtered["divergence_tier"] < 2]
if search:
    q = search.strip().lower()
    filtered = filtered[
        filtered["stock_code"].str.lower().str.contains(q)
        | filtered["name"].fillna("").str.lower().str.contains(q)
    ]

# Priority is the filter/divergence result, NOT the model -- divergence tier
# (0=double, 1=single, 2=none) sorts first; probability only breaks ties
# within a tier, exactly the "prioritas screening, probabilitas cuma
# pertimbangan" ordering the user asked for.
filtered = filtered.sort_values(
    ["divergence_tier", "probability"], ascending=[True, False], na_position="last",
).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total hasil filter", len(filtered))
c2.metric("🔥 Divergence ganda", int((filtered["divergence_tier"] == 0).sum()))
c3.metric("Divergence tunggal", int((filtered["divergence_tier"] == 1).sum()))

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan filter saat ini -- coba longgarkan rentang RSI atau status MACD.")
    st.stop()

st.subheader(f"📋 {len(filtered)} Saham -- diurutkan divergence dulu, probabilitas Swing sebagai tiebreaker")
st.caption("Klik satu baris untuk buka halaman detail saham itu.")

table_df = filtered.copy()
table_df["divergence_label"] = table_df.apply(divergence_detail, axis=1)
table_df["momentum"] = table_df.apply(lambda r: momentum_label(r["macd_hist"], r["macd_hist_slope_3d"]), axis=1)
table_df["money_flow"] = table_df["cmf_20"].apply(
    lambda v: "-" if pd.isna(v) else ("Akumulasi" if v > 0 else "Distribusi")
)
table_df["rvol_display"] = table_df["rvol_20"].apply(lambda v: "-" if pd.isna(v) else f"{v:.1f}x")
table_df["probability_pct"] = table_df["probability"].astype(float) * 100

display_cols = [
    "stock_code", "name", "close", "rsi_14", "macd_status", "momentum",
    "money_flow", "rvol_display", "divergence_label", "probability_pct", "regime",
]

event = st.dataframe(
    table_df[display_cols].reset_index(drop=True),
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_config={
        "stock_code": st.column_config.TextColumn("Kode"),
        "name": st.column_config.TextColumn("Nama"),
        "close": st.column_config.NumberColumn("Harga", format="%.0f"),
        "rsi_14": st.column_config.NumberColumn("RSI", format="%.1f"),
        "macd_status": st.column_config.TextColumn("Status MACD"),
        "momentum": st.column_config.TextColumn("Momentum Histogram"),
        "money_flow": st.column_config.TextColumn("Money Flow"),
        "rvol_display": st.column_config.TextColumn("Volume Relatif"),
        "divergence_label": st.column_config.TextColumn("Divergence"),
        "probability_pct": st.column_config.ProgressColumn("Probabilitas Swing", format="%.1f%%", min_value=0.0, max_value=100.0),
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

st.subheader("🏆 Prioritas Teratas")
top = filtered.head(9)
CARDS_PER_ROW = 3
rows = [top.iloc[i:i + CARDS_PER_ROW] for i in range(0, len(top), CARDS_PER_ROW)]
for row_chunk in rows:
    cols = st.columns(CARDS_PER_ROW)
    for col, (_, r) in zip(cols, row_chunk.iterrows()):
        with col:
            name = r["stock_code"] if pd.isna(r["name"]) else r["name"]
            div_badge = badge_html(DIVERGENCE_LABELS[r["divergence_tier"]], DIVERGENCE_COLORS[r["divergence_tier"]])
            macd_badge = badge_html(r["macd_status"], MACD_STATUS_COLORS.get(r["macd_status"], TEXT_MUTED))
            prob_txt = f"{float(r['probability']) * 100:.1f}%" if pd.notna(r["probability"]) else "belum ada prediksi"
            st.markdown(
                f"""
                <div class="mystocks-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div class="mystocks-ticker">{r['stock_code']}</div>
                            <div class="mystocks-muted">{name}</div>
                        </div>
                        {div_badge}
                    </div>
                    <div style="margin-top:0.7rem;">
                        <span class="mystocks-muted" style="font-size:0.72rem;">MACD</span> {macd_badge}
                        &nbsp;&nbsp;
                        <span class="mystocks-muted" style="font-size:0.72rem;">Regime</span> {regime_badge(r['regime'])}
                    </div>
                    <div style="margin-top:0.6rem;" class="mystocks-muted">
                        RSI {r['rsi_14']:.1f} &middot; {momentum_label(r['macd_hist'], r['macd_hist_slope_3d'])} &middot;
                        {"Akumulasi" if pd.notna(r['cmf_20']) and r['cmf_20'] > 0 else "Distribusi" if pd.notna(r['cmf_20']) else "-"}
                    </div>
                    <div style="margin-top:0.4rem;" class="mystocks-muted">Probabilitas Swing: {prob_txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Lihat Detail →", key=f"detail_{r['stock_code']}", width="stretch"):
                st.session_state["selected_ticker"] = r["stock_code"]
                st.switch_page("pages/1_📈_Detail_Saham.py")
            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
