import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.auth import is_logged_in, require_login
from app.data import load_data_freshness, load_dividend_screener_data, load_liquidity, load_live_prices, load_stock_list, load_suspended_tickers
from app.positions import has_active_position, mark_position
from app.style import data_freshness_note, format_traded_value, inject_base_css, liquidity_sidebar_filter, render_developer_footer
from features.dividend_screener import (
    DIVIDEND_YIELD_MIN_PCT,
    LOOKAHEAD_MONTHS,
    MONTH_NAMES_ID,
    build_dividend_table,
    filter_seasonally_upcoming,
)

st.set_page_config(page_title="MyStocks — Dividen Momentum", page_icon="💰", layout="wide")
inject_base_css()
require_login("Dividen Momentum")
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("💰 Dividen Momentum")
st.caption(
    "Dua cara melihat saham dividen: mana yang bayar PALING BESAR, dan mana yang historisnya "
    "bayar sekitar sekarang -- jadi kemungkinan akan bayar lagi dalam waktu dekat."
)
with st.expander("ℹ️ Catatan tentang jadwal cum-date (klik untuk buka)"):
    st.warning(
        "**Tidak ada jadwal cum-date resmi ke depan yang bisa didapat otomatis** -- sudah dicek "
        "langsung: field `exDividendDate` yfinance ternyata melaporkan tanggal TERAKHIR YANG SUDAH "
        "LEWAT, bukan yang akan datang (terbukti: BBCA menunjukkan 31 Agt 2026 padahal itu sudah 3+ "
        "minggu lalu saat dicek). Situs resmi BEI diblokir Cloudflare, dan API RapidAPI yang sudah "
        "dipakai di sini tidak punya endpoint kalender dividen. Semua yang ada di halaman ini dihitung "
        "dari **histori pembayaran dividen nyata** (`price_history.dividends`) -- tab kedua adalah "
        "**perkiraan berdasarkan pola tahun-tahun sebelumnya, BUKAN jadwal pasti**. Cum-date "
        "sebenarnya bisa berbeda atau tidak terjadi sama sekali tahun ini -- selalu cek pengumuman "
        "resmi RUPS/keterbukaan informasi sebelum mengambil keputusan.",
        icon="⚠️",
    )
data_freshness_note(load_data_freshness())

stocks_df = load_stock_list()
suspended_tickers = load_suspended_tickers()
panel = load_dividend_screener_data()

with st.sidebar:
    st.header("🔎 Filter")
    search = st.text_input("Cari kode/nama saham", placeholder="mis. BBCA atau bank")
    min_liq = liquidity_sidebar_filter("dividend_liq")


def _prep_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.merge(stocks_df, left_on="stock_code", right_on="code", how="left")
    if suspended_tickers:
        out = out[~out["stock_code"].isin(suspended_tickers)]
    out = out.merge(load_liquidity(), on="stock_code", how="left")
    if min_liq is not None:
        out = out[out["avg_traded_value"].fillna(0) >= min_liq]
    if search:
        q = search.strip().lower()
        out = out[
            out["stock_code"].str.lower().str.contains(q)
            | out["name"].fillna("").str.lower().str.contains(q)
        ]
    return out.reset_index(drop=True)


def _render_table(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        st.info("Tidak ada saham yang cocok dengan filter saat ini.")
        return
    table = df.copy()
    table["payout_display"] = table["payout_ratio"].apply(lambda v: f"{v:.0f}%" if pd.notna(v) else "-")
    table["last_div_display"] = table.apply(
        lambda r: (
            f"{r['last_dividend_date'].day} {MONTH_NAMES_ID[r['last_dividend_date'].month]} "
            f"{r['last_dividend_date'].year} (Rp{r['last_dividend_amount']:,.0f})"
        ).replace(",", "."),
        axis=1,
    )
    table["liq_display"] = table["avg_traded_value"].apply(format_traded_value)

    display_cols = [
        "stock_code", "name", "dividend_yield_pct", "payments_this_year_count", "payments_this_year_display",
        "payments_last_year_count", "payments_last_year_display", "last_div_display", "payout_display", "liq_display",
    ]
    event = st.dataframe(
        table[display_cols],
        width="stretch",
        hide_index=True,
        height=min(36 * (len(table) + 1) + 3, 500),
        column_config={
            "stock_code": st.column_config.TextColumn("Kode"),
            "name": st.column_config.TextColumn("Nama"),
            "dividend_yield_pct": st.column_config.NumberColumn("Dividend Yield", format="%.1f%%"),
            "payments_this_year_count": st.column_config.NumberColumn("Bayar Tahun Ini", format="%d"),
            "payments_this_year_display": st.column_config.TextColumn("Bulan & Persen Tahun Ini"),
            "payments_last_year_count": st.column_config.NumberColumn("Bayar Tahun Lalu", format="%d"),
            "payments_last_year_display": st.column_config.TextColumn("Bulan & Persen Tahun Lalu"),
            "last_div_display": st.column_config.TextColumn("Dividen Terakhir"),
            "payout_display": st.column_config.TextColumn("Payout Ratio"),
            "liq_display": st.column_config.TextColumn("Transaksi/hari (rata2 60h)"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        st.session_state["selected_ticker"] = table.iloc[selected_rows[0]]["stock_code"]
        st.switch_page("pages/1_📈_Detail_Saham.py")

    # Tandai Beli: tabel ini tidak punya harga/stop/target saat ini (cuma
    # histori tanggal bayar dividen) -- harga beli default diambil live,
    # stop/target tetap diisi manual sama seperti halaman screener lain.
    with st.expander("📌 Tandai saham dari tabel di atas sebagai sudah dibeli"):
        div_mark_options = {
            f"{r['stock_code']} — {r['name'] if pd.notna(r['name']) else r['stock_code']}": r
            for _, r in table.iterrows()
        }
        div_mark_label = st.selectbox("Pilih saham", list(div_mark_options.keys()), key=f"{key}_mark_select")
        div_mark_row = div_mark_options[div_mark_label]
        if not is_logged_in():
            st.caption("🔒 Login untuk menandai saham sebagai sudah dibeli.")
        elif has_active_position(st.user.email, div_mark_row["stock_code"]):
            st.caption(f"📌 {div_mark_row['stock_code']} sudah ditandai sebagai posisi aktif.")
        else:
            live = load_live_prices((div_mark_row["stock_code"],))
            div_entry_default = float(live.get(div_mark_row["stock_code"], 0.0))
            m1, m2, m3 = st.columns(3)
            div_entry = m1.number_input("Harga Beli", min_value=0.0, value=div_entry_default, step=1.0, key=f"{key}_mark_entry")
            div_stop = m2.number_input("Stop Loss", min_value=0.0, value=round(div_entry_default * 0.975, 1), step=1.0, key=f"{key}_mark_stop")
            div_target = m3.number_input("Take Profit", min_value=0.0, value=round(div_entry_default * 1.05, 1), step=1.0, key=f"{key}_mark_target")
            if st.button("📌 Tandai Beli", key=f"{key}_mark_btn"):
                div_context = {
                    "dividend_yield_pct": round(float(div_mark_row["dividend_yield_pct"]), 2) if pd.notna(div_mark_row["dividend_yield_pct"]) else None,
                    "dividen_terakhir": div_mark_row["last_div_display"],
                    "payout_ratio": div_mark_row["payout_display"],
                }
                mark_position(
                    st.user.email, div_mark_row["stock_code"], "dividend_screener_v1",
                    div_entry, div_stop, div_target,
                    entry_source="dividen_momentum", entry_context=json.dumps(div_context),
                )
                st.success(f"{div_mark_row['stock_code']} ditandai. Lihat halaman **Posisi Saya**.")
                st.rerun()


tab_biggest, tab_upcoming = st.tabs(["🏆 Dividen Terbesar", "📅 Akan Bayar dalam Waktu Dekat"])

with tab_biggest:
    st.caption("Semua saham dengan data dividend yield, diurutkan dari yang terbesar -- tanpa syarat waktu.")
    min_yield_biggest = st.slider(
        "Yield minimum", min_value=0.0, max_value=25.0, value=0.0, step=0.5,
        format="%.1f%%", key="min_yield_biggest",
    )
    base_biggest = build_dividend_table(panel, min_yield_pct=min_yield_biggest)
    table_biggest = _prep_table(base_biggest)
    st.metric("Jumlah saham", len(table_biggest) if not table_biggest.empty else 0)
    _render_table(table_biggest, key="dividend_biggest_select")

with tab_upcoming:
    st.caption(
        f"Yield ≥ ambang di bawah, DAN historisnya pernah bayar dividen di bulan ini atau beberapa "
        "bulan ke depan (atur di bawah)."
    )
    c1, c2 = st.columns(2)
    min_yield_upcoming = c1.slider(
        "Yield minimum", min_value=0.0, max_value=25.0, value=DIVIDEND_YIELD_MIN_PCT, step=0.5,
        format="%.1f%%", key="min_yield_upcoming",
    )
    lookahead = c2.slider(
        "Berapa bulan ke depan dianggap 'dalam waktu dekat'", min_value=1, max_value=3,
        value=LOOKAHEAD_MONTHS, key="lookahead_months",
    )
    base_upcoming = build_dividend_table(panel, min_yield_pct=min_yield_upcoming)
    upcoming = filter_seasonally_upcoming(base_upcoming, lookahead_months=lookahead)
    table_upcoming = _prep_table(upcoming)
    st.metric("Jumlah saham", len(table_upcoming) if not table_upcoming.empty else 0)
    _render_table(table_upcoming, key="dividend_upcoming_select")
