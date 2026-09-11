import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.auth import require_login
from app.data import (
    load_data_freshness,
    load_latest_predictions,
    load_liquidity,
    load_screener_raw_panel,
    load_stock_list,
    load_suspended_tickers,
)
from app.style import (
    ACCENT,
    badge_html,
    data_freshness_note,
    format_traded_value,
    inject_base_css,
    liquidity_sidebar_filter,
    regime_badge,
    render_developer_footer,
)
from features.momentum_screener import compute_screener_panel

st.set_page_config(page_title="MyStocks — Rekomendasi Emitten", page_icon="🏆", layout="wide")
inject_base_css()
require_login("Rekomendasi Emitten")
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🏆 Rekomendasi Emitten")
st.caption(
    "Menggabungkan dua alat screening yang independen satu sama lain -- Swing (model ML, "
    "horizon 10 hari) dan Momentum Screener (aturan teknikal tervalidasi lewat backtest) -- "
    "untuk mencari saham yang mendapat sinyal dari KEDUANYA pada saat yang sama, bukan cuma "
    "dari satu sudut pandang."
)
st.success(
    "**Bukti historis (backtest 5 tahun, `scripts/search_momentum_rules.py` + lanjutannya)**: saham "
    "yang lolos Sinyal Tervalidasi Momentum Screener SENDIRIAN (termasuk kriteria Anchored VWAP, "
    "lihat halaman Momentum Screener) menang **42,4%** dari kejadian (n=523, batas bawah keyakinan "
    "95%: 38,3%) -- satu-satunya angka gabungan yang sudah diuji ketat lewat walk-forward validation "
    "di halaman ini. Menambahkan syarat Swing ≥WATCH di atasnya BELUM diuji ulang secara terpisah "
    "untuk kombinasi dua-alat ini -- anggap sebagai penyaring tambahan yang masuk akal, bukan angka "
    "yang sudah terbukti sendiri.",
    icon="🏆",
)
st.info(
    "Halaman ini murni **menyaring & menggabungkan** hasil dari dua halaman lain -- tidak ada "
    "perhitungan baru. Probabilitas Swing persis sama dengan yang tampil di halamannya sendiri; "
    "status Momentum Screener persis sama dengan kolom ✅ Tervalidasi di sana.\n\n"
    "**Catatan**: halaman Turnaround (model 6-bulan) sudah dipensiunkan -- backtest top-2 "
    "(`scripts/compare_turnaround_v2_top2.py`) menunjukkan performanya, dan bahkan usulan "
    "penggantinya, keduanya jauh di bawah Swing untuk tujuan jangka pendek/menengah manapun -- "
    "jadi tidak lagi ikut dihitung di sini.",
    icon="ℹ️",
)
data_freshness_note(load_data_freshness())

TIER_LABELS = {2: "🌟 Keduanya Sepakat (2/2)", 1: "Salah Satu (1/2)"}
TIER_COLORS = {2: "#FBBF24", 1: ACCENT}


def fmt_pct(value) -> str:
    return "-" if pd.isna(value) else f"{float(value) * 100:.1f}%"


with st.spinner("Menggabungkan hasil Swing dan Momentum Screener..."):
    swing = load_latest_predictions()
    raw_panel = load_screener_raw_panel(lookback_days=60)
    momentum = compute_screener_panel(raw_panel)
    stocks_df = load_stock_list()

if stocks_df.empty:
    st.warning("Belum ada data saham. Jalankan pipeline terlebih dahulu.")
    st.stop()

base = stocks_df.rename(columns={"code": "stock_code"})
df = base.merge(
    swing[["stock_code", "decision", "probability"]].rename(
        columns={"decision": "swing_decision", "probability": "swing_prob"}
    ) if not swing.empty else pd.DataFrame(columns=["stock_code", "swing_decision", "swing_prob"]),
    on="stock_code", how="left",
)
df = df.merge(
    momentum[["stock_code", "validated_signal", "regime", "macd_status", "close", "rsi_14"]]
    if not momentum.empty else pd.DataFrame(columns=["stock_code", "validated_signal", "regime", "macd_status", "close", "rsi_14"]),
    on="stock_code", how="left",
)

# swing_hit uses WATCH-or-better (probability >= base_rate, i.e. NOT AVOID) --
# matches what scripts/backtest_triple_intersection.py originally tested;
# a full Swing BUY was found to essentially never co-occur with the
# Momentum-validated bottoming regime historically (n=0), so requiring
# BUY specifically here would empty this page out.
df["swing_hit"] = df["swing_decision"].isin(["BUY", "WATCH"])
df["momentum_hit"] = df["validated_signal"].fillna(False)
df["agreement_count"] = df[["swing_hit", "momentum_hit"]].sum(axis=1).astype(int)
df = df.merge(load_liquidity(), on="stock_code", how="left")

# Suspended stocks (frozen quote, zero volume) excluded from ranking entirely --
# see app.data.load_suspended_tickers's docstring.
suspended_tickers = load_suspended_tickers()
n_suspended_hidden = int(df["stock_code"].isin(suspended_tickers).sum())
if suspended_tickers:
    df = df[~df["stock_code"].isin(suspended_tickers)]

with st.sidebar:
    st.header("🔎 Filter")
    min_agreement = st.radio(
        "Minimal kesepakatan", [1, 2],
        format_func=lambda n: {1: "1 dari 2 (semua yang dapat sinyal)", 2: "Keduanya (2 dari 2, paling ketat)"}[n],
        index=1,
    )
    search = st.text_input("Cari kode/nama saham", placeholder="mis. CYBR atau bank")
min_liq = liquidity_sidebar_filter("rekomendasi_liq")

filtered = df[df["agreement_count"] >= min_agreement].copy()
if min_liq is not None:
    filtered = filtered[filtered["avg_traded_value"].fillna(0) >= min_liq]
if search:
    q = search.strip().lower()
    filtered = filtered[
        filtered["stock_code"].str.lower().str.contains(q)
        | filtered["name"].fillna("").str.lower().str.contains(q)
    ]

filtered = filtered.sort_values(
    ["agreement_count", "swing_prob"], ascending=[False, False], na_position="last",
).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total ditampilkan", len(filtered))
c2.metric("🌟 Keduanya Sepakat (2/2)", int((df["agreement_count"] == 2).sum()))
c3.metric("1 dari 2", int((df["agreement_count"] == 1).sum()))
if n_suspended_hidden:
    st.caption(
        f"🚫 {n_suspended_hidden} saham disembunyikan dari daftar karena tampak sedang disuspend "
        "(harga beku, volume nol beberapa hari terakhir)."
    )

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan filter saat ini -- coba turunkan minimal kesepakatan.")
    st.stop()

st.subheader(f"📋 {len(filtered)} Saham -- diurutkan tingkat kesepakatan dulu")
st.caption("Klik satu baris untuk buka halaman detail saham itu.")

table_df = filtered.copy()
table_df["tingkat"] = table_df["agreement_count"].map(TIER_LABELS)
table_df["swing_display"] = table_df.apply(
    lambda r: "-" if pd.isna(r["swing_decision"]) else f"{r['swing_decision']} ({fmt_pct(r['swing_prob'])})", axis=1,
)
table_df["momentum_display"] = table_df["momentum_hit"].apply(lambda v: "✅ Ya" if v else "-")
table_df["liq_display"] = table_df["avg_traded_value"].apply(format_traded_value)

display_cols = ["stock_code", "name", "tingkat", "close", "swing_display", "momentum_display", "regime", "liq_display"]

event = st.dataframe(
    table_df[display_cols].reset_index(drop=True),
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_config={
        "stock_code": st.column_config.TextColumn("Kode"),
        "name": st.column_config.TextColumn("Nama"),
        "tingkat": st.column_config.TextColumn("Tingkat Konsensus"),
        "close": st.column_config.NumberColumn("Harga", format="%.0f"),
        "swing_display": st.column_config.TextColumn("Swing"),
        "momentum_display": st.column_config.TextColumn("Momentum"),
        "regime": st.column_config.TextColumn("Regime"),
        "liq_display": st.column_config.TextColumn("Transaksi/hari (rata2 60h)"),
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
            tier_badge = badge_html(TIER_LABELS[r["agreement_count"]], TIER_COLORS[r["agreement_count"]])
            swing_txt = "-" if pd.isna(r["swing_decision"]) else f"{r['swing_decision']} · {fmt_pct(r['swing_prob'])}"
            momentum_txt = "✅ Tervalidasi" if r["momentum_hit"] else "-"
            # dedent() strips this f-string's ~16-space Python source
            # indentation, and the blank-line filter drops any line that
            # collapses to pure whitespace once its {} is substituted --
            # both load-bearing, not cosmetic. See the full "why" (a real
            # screenshot bug, not a guess) at the identical card-rendering
            # fix in app/pages/4_📡_Momentum_Screener.py: without dedent(),
            # Markdown reads a 4+-space-indented "<div..." line as an
            # indented CODE block rather than HTML; without the blank-line
            # filter, any interpolated value that goes empty splits the
            # block at that blank line and everything after it (still
            # carrying its own nested indentation) falls into the same trap.
            # min-height on the name reserves room for a 2-line company name
            # (e.g. "PT MNC Digital Entertainment Tbk") so a row where only
            # ONE card has a long name doesn't end up taller than its
            # siblings -- caught from a real screenshot: MSIN's card grew
            # past BUKA/RATU's in the same row, throwing off their "Lihat
            # Detail" buttons below.
            card_html = textwrap.dedent(f"""
                <div class="mystocks-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div class="mystocks-ticker">{r['stock_code']}</div>
                            <div class="mystocks-muted" style="min-height:2.6em; line-height:1.3em;">{name}</div>
                        </div>
                        {tier_badge}
                    </div>
                    <div style="margin-top:0.6rem;">{regime_badge(r['regime'])}</div>
                    <div style="margin-top:0.6rem;" class="mystocks-muted">
                        <b>Swing</b>: {swing_txt}<br>
                        <b>Momentum</b>: {momentum_txt}<br>
                        <b>Transaksi/hari</b>: {format_traded_value(r.get('avg_traded_value'))}
                    </div>
                </div>
                """)
            card_html = "\n".join(line for line in card_html.splitlines() if line.strip())
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button("Lihat Detail →", key=f"detail_{r['stock_code']}", width="stretch"):
                st.session_state["selected_ticker"] = r["stock_code"]
                st.switch_page("pages/1_📈_Detail_Saham.py")
            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
