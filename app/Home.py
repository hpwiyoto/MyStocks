import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from app.data import load_latest_predictions, load_live_prices
from app.style import decision_badge, inject_base_css, regime_badge, render_developer_footer
from engine.predict import run as predict_run
from features.build_features import run as build_features_run
from pipeline.ingest_price import run as ingest_price_run

st.set_page_config(page_title="MyStocks — Screener", page_icon="📈", layout="wide")
inject_base_css()

ENTRY_RANGE_PCT = 0.005  # +-0.5% zona beli di sekitar harga saat ini, bukan satu angka persis
ENTRY_RANGE_MIN_RUPIAH = 2  # +-1% saham gocap (Rp50) < Rp1 -- dibulatkan jadi "50 - 50", jaminan lebar minimum
TABLE_TOP_N = 25  # cap "mode semua" (tanpa daftar ticker manual) -- tabel, live-price overlay, & tombol
                  # update harga sama-sama dibatasi ke ini. Naik dari 15 atas permintaan eksplisit.
MANUAL_TICKER_MAX = 40  # sanity cap kalau user comma-paste daftar ticker yang sangat panjang --
                         # jaga biaya live-price fetch & tombol update harga tetap wajar


def entry_range(price: float) -> tuple[float, float]:
    half_width = max(price * ENTRY_RANGE_PCT, ENTRY_RANGE_MIN_RUPIAH)
    return price - half_width, price + half_width

st.title("📈 MyStocks Screener")
st.caption("Prediksi harian saham IDX — probabilitas naik ≥5% sebelum stop-loss -2.5% dalam 10 hari trading.")

# Home.py uses the classic file-based pages/ structure, where -- unlike the
# newer st.navigation API -- widget-keyed session_state is NOT reliably kept
# across a page switch (confirmed by testing: filters reset even with a
# plain `key=` set). Persisting the actual value in a plain (non-widget)
# session_state slot and feeding it back in as value=/default=/index= on
# every rerun works around that; a widget `key=` alone does not.
def _persisted(name, default):
    return st.session_state.get(f"persist_{name}", default)


def _save_persisted(name, value):
    st.session_state[f"persist_{name}"] = value


PRICE_FILTER_OPTIONS = ["Semua", "Di bawah 50", "50 - 100", "100 - 1.000", "Di atas 1.000"]

with st.sidebar:
    st.header("🔎 Filter")
    search = st.text_input(
        "Cari kode/nama saham",
        placeholder="mis. BBCA atau bank, atau BBCA, TLKM, ASII untuk beberapa ticker",
        help="Pisahkan dengan koma untuk menampilkan beberapa ticker tertentu sekaligus "
             "(cocok kode persis, tidak dibatasi Top 25).",
        value=_persisted("search", ""),
    )
    _save_persisted("search", search)

    decision_filter = st.multiselect(
        "Keputusan", ["BUY", "WATCH", "AVOID"],
        default=_persisted("decision_filter", ["BUY", "WATCH", "AVOID"]),
    )
    _save_persisted("decision_filter", decision_filter)

    min_prob = st.slider("Probabilitas minimum", 0.0, 1.0, _persisted("min_prob", 0.0), 0.05)
    _save_persisted("min_prob", min_prob)

    _persisted_price = _persisted("price_filter", "Semua")
    price_filter = st.selectbox(
        "Harga saham", PRICE_FILTER_OPTIONS,
        index=PRICE_FILTER_OPTIONS.index(_persisted_price) if _persisted_price in PRICE_FILTER_OPTIONS else 0,
    )
    _save_persisted("price_filter", price_filter)

with st.spinner("Memuat prediksi terbaru..."):
    df = load_latest_predictions()

if df.empty:
    st.warning(
        "Belum ada data prediksi. Jalankan `pipeline.ingest_price` → `features.build_features` "
        "→ `engine.predict` terlebih dahulu."
    )
    st.stop()

regime_options = sorted(df["regime"].dropna().unique().tolist())
# drop persisted selections that no longer exist in today's data (e.g.
# after a Refresh) -- multiselect errors if default holds a value not in
# the current options list
_persisted_regime = [r for r in _persisted("regime_filter", regime_options) if r in regime_options]
with st.sidebar:
    regime_filter = st.multiselect("Regime", regime_options, default=_persisted_regime)
_save_persisted("regime_filter", regime_filter)

render_developer_footer()

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
# Comma present -> treat as a manual list of exact ticker codes (the user
# typed specific stocks they want, e.g. "BBCA, TLKM, ASII") rather than one
# substring query -- a single term still matches by substring against both
# code and name (so "bank" or a partial code keeps working as before).
manual_tickers = [t.strip().lower() for t in search.split(",") if t.strip()] if search else []
manual_ticker_mode = len(manual_tickers) > 1
if manual_ticker_mode:
    filtered = filtered[filtered["stock_code"].str.lower().isin(manual_tickers)]
elif search:
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

# Refreshing all ~900 tickers (fetch + features + predict) takes tens of
# minutes even with the fast incremental feature path -- far too slow for a
# synchronous button click. Scoping it to the same top-N shown in the table
# below (not the full filtered set, which could still be hundreds) keeps
# every click cheap regardless of how broad the sidebar filters are.
# Manual ticker mode is the one exception: the user explicitly named these
# stocks, so show every one of them (up to the MANUAL_TICKER_MAX safety cap)
# instead of truncating to the "best N by probability" ranking, which would
# silently drop a ticker they specifically searched for.
table_source = filtered.head(MANUAL_TICKER_MAX if manual_ticker_mode else TABLE_TOP_N).copy()
refresh_codes = table_source["stock_code"].tolist()

# Live price overlay: on every page load/rerun (not just the button below),
# fetch a CURRENT quote for just these on-screen tickers (<=25 by default,
# <=MANUAL_TICKER_MAX in manual-search mode) and use it in place of
# `entry_price` (which is only as fresh as the last daily pipeline run, i.e.
# up to a day stale). Cheap enough (~0.3s/ticker, cached 30s) to run
# unconditionally here -- this is why the on-screen count is capped at all,
# so this stays fast even though it runs on every rerun, not just on click.
live_prices = load_live_prices(tuple(refresh_codes))
if live_prices:
    table_source["entry_price"] = table_source.apply(
        lambda r: live_prices.get(r["stock_code"], r["entry_price"]), axis=1,
    )
if refresh_codes:
    n_live = len(live_prices)
    st.caption(
        f"💹 Harga live: {n_live}/{len(refresh_codes)} ticker berhasil diambil real-time "
        f"(sisanya pakai harga penutupan terakhir)." if n_live < len(refresh_codes)
        else f"💹 Harga live untuk {n_live} ticker yang ditampilkan (update tiap 30 detik)."
    )

with st.sidebar:
    st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
    if st.button(f"🔄 Update harga ({len(refresh_codes)} ticker tampil)", width="stretch"):
        if not refresh_codes:
            st.sidebar.warning("Tidak ada ticker yang cocok filter saat ini untuk di-update.")
        else:
            with st.spinner(f"Mengambil harga terbaru untuk {len(refresh_codes)} ticker..."):
                ingest_price_run(tickers=refresh_codes)
                build_features_run(tickers=refresh_codes)
                predict_run(tickers=refresh_codes)
            st.cache_data.clear()
            st.rerun()

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan filter saat ini.")
    st.stop()

# --- Top peluang: highlight kartu untuk beberapa probabilitas tertinggi ---
TOP_N_CARDS = 9
top = table_source.head(TOP_N_CARDS)  # already has live prices overlaid, see above
if manual_ticker_mode:
    st.subheader(f"🏆 {len(top)} Ticker yang Dicari")
else:
    st.subheader(f"🏆 Top {min(TOP_N_CARDS, len(top))} Peluang Tertinggi")

CARDS_PER_ROW = 3
rows = [top.iloc[i : i + CARDS_PER_ROW] for i in range(0, len(top), CARDS_PER_ROW)]
for row_chunk in rows:
    cols = st.columns(CARDS_PER_ROW)
    for col, (_, r) in zip(cols, row_chunk.iterrows()):
        with col:
            # NULL from SQL comes back as float NaN, not None -- and NaN is
            # truthy in Python, so `r["name"] or r["stock_code"]` would
            # silently print "nan" instead of falling back (found and fixed
            # for real on Detail Saham's sector panel; guarded here too).
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
            st.progress(min(max(float(r["probability"]), 0.0), 1.0), text=f"Probabilitas: {prob_pct:.1f}%")
            current_price = float(r["entry_price"])
            entry_low, entry_high = entry_range(current_price)
            e1, e2 = st.columns(2)
            e1.markdown(f"<span class='mystocks-muted'>Harga Saat Ini</span><br>{current_price:,.0f}", unsafe_allow_html=True)
            e2.markdown(f"<span class='mystocks-muted'>Entry</span><br>{entry_low:,.0f} - {entry_high:,.0f}", unsafe_allow_html=True)
            e3, e4 = st.columns(2)
            e3.markdown(f"<span class='mystocks-muted'>Stop Loss</span><br>{float(r['stop_loss_price']):,.0f}", unsafe_allow_html=True)
            e4.markdown(f"<span class='mystocks-muted'>Take Profit</span><br>{float(r['take_profit_price']):,.0f}", unsafe_allow_html=True)
            if st.button("Lihat Detail →", key=f"detail_{r['stock_code']}", width="stretch"):
                st.session_state["selected_ticker"] = r["stock_code"]
                st.switch_page("pages/1_📈_Detail_Saham.py")
            st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Tabel ranking: dibatasi top N teratas (bukan seluruh hasil filter) --
# menampilkan ratusan baris sekaligus memberatkan render browser & query
# tanpa manfaat nyata, karena yang dicari selalu peluang terbaik dulu.
# (table_source dihitung di atas, sekalian dipakai untuk live-price overlay
# & tombol Update Harga.)
if manual_ticker_mode:
    st.subheader(f"📋 {len(table_source)} Saham Dicari — urut berdasarkan probabilitas")
    if len(filtered) > MANUAL_TICKER_MAX:
        st.caption(
            f"Menampilkan {MANUAL_TICKER_MAX} dari {len(filtered)} ticker yang cocok (dibatasi supaya tetap ringan) "
            "-- persempit daftar ticker yang dicari untuk melihat semuanya."
        )
elif len(filtered) > TABLE_TOP_N:
    st.subheader(f"📋 Top {TABLE_TOP_N} Saham dari {len(filtered)} — urut berdasarkan probabilitas")
    st.caption(
        f"Menampilkan {TABLE_TOP_N} peluang probabilitas tertinggi saja (bukan semua {len(filtered)} hasil filter) "
        "supaya tetap ringan. Persempit filter di sidebar untuk melihat saham tertentu."
    )
else:
    st.subheader(f"📋 Semua Saham ({len(filtered)}) — urut berdasarkan probabilitas")
st.caption("Klik header kolom untuk sortir ulang. Klik satu baris untuk buka halaman detail saham itu.")

table_df = table_source[["stock_code", "name", "decision", "probability", "regime", "entry_price", "stop_loss_price", "take_profit_price"]].reset_index(drop=True)
table_df["probability"] = table_df["probability"].astype(float) * 100  # ProgressColumn format="%.1f%%" doesn't auto-scale from 0-1
table_df["entry_range"] = table_df["entry_price"].apply(lambda p: "{:,.0f} - {:,.0f}".format(*entry_range(p)))

event = st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_order=[
        "stock_code", "name", "decision", "probability", "regime",
        "entry_price", "entry_range", "stop_loss_price", "take_profit_price",
    ],
    column_config={
        "stock_code": st.column_config.TextColumn("Kode"),
        "name": st.column_config.TextColumn("Nama"),
        "decision": st.column_config.TextColumn("Keputusan"),
        "probability": st.column_config.ProgressColumn("Probabilitas", format="%.1f%%", min_value=0.0, max_value=100.0),
        "regime": st.column_config.TextColumn("Regime"),
        "entry_price": st.column_config.NumberColumn("Harga Saat Ini", format="%.0f"),
        "entry_range": st.column_config.TextColumn("Entry"),
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
