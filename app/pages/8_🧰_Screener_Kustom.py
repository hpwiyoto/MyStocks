import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.auth import require_login
from app.data import load_data_freshness, load_screener_universe, load_suspended_tickers
from app.style import (
    data_freshness_note,
    format_traded_value,
    inject_base_css,
    liquidity_sidebar_filter,
    render_developer_footer,
)
from features.custom_screener import NUMERIC_PARAMS, OPERATORS, PARAM_REGISTRY, apply_conditions, param_options_by_category

st.set_page_config(page_title="MyStocks — Screener Kustom", page_icon="🧰", layout="wide")
inject_base_css()
require_login("Screener Kustom")
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

st.title("🧰 Screener Kustom")
st.caption(
    "Susun sendiri kondisi filter dari parameter apa pun yang tersedia -- teknikal, "
    "fundamental, harga/likuiditas -- gaya screener Stockbit. Semua kondisi digabung dengan "
    "DAN (AND): saham harus lolos SEMUA kondisi yang Anda tambahkan. Ini murni alat "
    "eksplorasi manual -- BUKAN kombinasi yang sudah terbukti lewat backtest seperti "
    "'Sinyal Tervalidasi' di Momentum Screener."
)
data_freshness_note(load_data_freshness())

with st.spinner("Memuat data seluruh saham..."):
    universe = load_screener_universe()

if universe.empty:
    st.warning("Belum ada data fitur/harga yang cukup. Jalankan pipeline & features terlebih dahulu.")
    st.stop()

# Same exclusion as every other screener page -- see
# app.data.load_suspended_tickers's docstring (suspensi, streak ARA/ARB,
# dan Papan Pemantauan Khusus/FCA sekaligus).
suspended_tickers = load_suspended_tickers()
n_hidden = int(universe["stock_code"].isin(suspended_tickers).sum())
if suspended_tickers:
    universe = universe[~universe["stock_code"].isin(suspended_tickers)]

with st.sidebar:
    st.header("🔎 Filter Tambahan")
    search = st.text_input("Cari kode/nama saham", placeholder="mis. BBCA atau bank")
    min_liq = liquidity_sidebar_filter("custom_screener_liq")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

# --- Condition builder --------------------------------------------------
if "csk_cids" not in st.session_state:
    st.session_state["csk_cids"] = []
if "csk_next_id" not in st.session_state:
    st.session_state["csk_next_id"] = 1

GROUPED = param_options_by_category()
ALL_PARAM_KEYS = [k for group in GROUPED.values() for k in group]


def _param_label(key: str) -> str:
    meta = PARAM_REGISTRY[key]
    return f"{meta['category']} · {meta['label']}"


def _numeric_step(median: float) -> float:
    m = abs(median) if pd.notna(median) else 0
    if m >= 1000:
        return 10.0
    if m >= 100:
        return 1.0
    if m >= 10:
        return 0.5
    if m >= 1:
        return 0.1
    return 0.01


st.subheader("Kondisi Filter")

top1, top2 = st.columns([1, 5])
with top1:
    if st.button("+ Tambah Kondisi", width="stretch"):
        cid = st.session_state["csk_next_id"]
        st.session_state["csk_next_id"] += 1
        st.session_state["csk_cids"].append(cid)
        st.rerun()
with top2:
    if st.session_state["csk_cids"] and st.button("Hapus semua kondisi"):
        st.session_state["csk_cids"] = []
        st.rerun()

conditions: list[dict] = []

if not st.session_state["csk_cids"]:
    st.caption("Belum ada kondisi -- klik **+ Tambah Kondisi** untuk mulai menyaring. Tanpa kondisi, seluruh saham ditampilkan.")

for cid in list(st.session_state["csk_cids"]):
    row = st.container(border=True)
    with row:
        c_param, c_op, c_cmp, c_val, c_remove = st.columns([2.6, 1.2, 1.6, 2.8, 0.5])

        param_key = c_param.selectbox(
            "Parameter", ALL_PARAM_KEYS, format_func=_param_label, key=f"csk_param_{cid}",
            label_visibility="collapsed",
        )
        kind = PARAM_REGISTRY[param_key]["kind"]

        if kind == "numeric":
            operator = c_op.selectbox("Operator", OPERATORS, key=f"csk_op_{cid}", label_visibility="collapsed")
            compare_mode = False
            if operator != "antara":
                compare_mode = c_cmp.checkbox(
                    "🆚 vs parameter lain", key=f"csk_cmp_{cid}",
                    help="Bandingkan nilai parameter ini langsung ke parameter lain (mis. Close > SMA 20), bukan ke angka tetap.",
                )
            else:
                c_cmp.caption("(rentang, tidak bisa dibandingkan ke parameter lain)")

            if compare_mode:
                other_options = [k for k in NUMERIC_PARAMS if k != param_key]
                other_key = c_val.selectbox(
                    "Parameter pembanding", other_options, format_func=_param_label,
                    key=f"csk_cmpparam_{cid}", label_visibility="collapsed",
                )
                conditions.append({"param": param_key, "kind": "numeric", "operator": operator, "compare_to": other_key})
            elif operator == "antara":
                median = pd.to_numeric(universe[param_key], errors="coerce").median()
                q1 = pd.to_numeric(universe[param_key], errors="coerce").quantile(0.25)
                q3 = pd.to_numeric(universe[param_key], errors="coerce").quantile(0.75)
                default_lo = float(q1) if pd.notna(q1) else 0.0
                default_hi = float(q3) if pd.notna(q3) else 0.0
                step = _numeric_step(median if pd.notna(median) else 0.0)
                cv1, cv2 = c_val.columns(2)
                lo = cv1.number_input("Min", value=round(default_lo, 4), step=step, key=f"csk_min_{cid}_{param_key}")
                hi = cv2.number_input("Maks", value=round(default_hi, 4), step=step, key=f"csk_max_{cid}_{param_key}")
                conditions.append({"param": param_key, "kind": "numeric", "operator": "antara", "value": (lo, hi)})
            else:
                median = pd.to_numeric(universe[param_key], errors="coerce").median()
                default_val = float(median) if pd.notna(median) else 0.0
                step = _numeric_step(default_val)
                val = c_val.number_input(
                    "Nilai", value=round(default_val, 4), step=step, key=f"csk_val_{cid}_{param_key}",
                    label_visibility="collapsed",
                )
                conditions.append({"param": param_key, "kind": "numeric", "operator": operator, "value": val})

        elif kind == "boolean":
            c_op.caption("=")
            choice = c_val.radio(
                "Nilai", ["Ya", "Tidak"], key=f"csk_bool_{cid}_{param_key}", horizontal=True,
                label_visibility="collapsed",
            )
            conditions.append({"param": param_key, "kind": "boolean", "value": choice == "Ya"})

        else:  # categorical
            c_op.caption("salah satu dari")
            options = sorted(universe[param_key].dropna().unique().tolist())
            chosen = c_val.multiselect(
                "Nilai", options, default=options, key=f"csk_cat_{cid}_{param_key}",
                label_visibility="collapsed",
            )
            conditions.append({"param": param_key, "kind": "categorical", "value": chosen})

        if c_remove.button("🗑️", key=f"csk_remove_{cid}"):
            st.session_state["csk_cids"].remove(cid)
            st.rerun()

filtered = apply_conditions(universe, conditions)

if min_liq is not None:
    filtered = filtered[filtered["avg_traded_value"].fillna(0) >= min_liq]

if search:
    q = search.strip().lower()
    filtered = filtered[
        filtered["stock_code"].str.lower().str.contains(q)
        | filtered["name"].fillna("").str.lower().str.contains(q)
    ]

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)

sort_col1, sort_col2, sort_col3 = st.columns([2.5, 1.5, 2])
sort_key = sort_col1.selectbox(
    "Urutkan berdasarkan", ["probability"] + ALL_PARAM_KEYS, format_func=lambda k: "Probabilitas Swing" if k == "probability" else _param_label(k),
    key="csk_sort_key",
)
sort_desc = sort_col2.checkbox("Urutan turun (desc)", value=True, key="csk_sort_desc")

if sort_key in filtered.columns:
    filtered = filtered.sort_values(sort_key, ascending=not sort_desc, na_position="last").reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total hasil filter", len(filtered))
c2.metric("Total dipantau", len(universe))
c3.metric("Kondisi aktif", len(conditions))
if n_hidden:
    st.caption(f"🚫 {n_hidden} saham disembunyikan (suspend/ARA-ARB/Papan Pemantauan Khusus) -- lihat `app.data.load_suspended_tickers`.")

if filtered.empty:
    st.info("Tidak ada saham yang cocok dengan kondisi saat ini -- coba longgarkan salah satu kondisi.")
    st.stop()

# Show base identity/price columns plus every column actually referenced by
# an active condition (both the primary param and its compare-to param, if
# any) -- so the user can see WHY a row matched, not just that it did.
base_cols = ["stock_code", "name", "close", "avg_traded_value", "probability"]
referenced = []
for cond in conditions:
    for k in (cond.get("param"), cond.get("compare_to")):
        if k and k not in base_cols and k not in referenced:
            referenced.append(k)
display_cols = base_cols + referenced

table_df = filtered.copy()
table_df["liq_display"] = table_df["avg_traded_value"].apply(format_traded_value)
table_df["probability_pct"] = table_df["probability"].astype(float) * 100

column_config = {
    "stock_code": st.column_config.TextColumn("Kode"),
    "name": st.column_config.TextColumn("Nama"),
    "close": st.column_config.NumberColumn("Harga", format="%.0f"),
    "avg_traded_value": None,
    "liq_display": st.column_config.TextColumn("Transaksi/hari (60h)"),
    "probability": None,
    "probability_pct": st.column_config.ProgressColumn("Probabilitas Swing", format="%.1f%%", min_value=0.0, max_value=100.0),
}
final_display_cols = [c for c in display_cols if c not in ("avg_traded_value", "probability")] + ["liq_display", "probability_pct"]
for col in referenced:
    if col not in column_config:
        meta = PARAM_REGISTRY.get(col, {})
        column_config[col] = st.column_config.NumberColumn(meta.get("label", col), format="%.4g") if meta.get("kind") == "numeric" else st.column_config.TextColumn(meta.get("label", col))

st.subheader(f"📋 {len(filtered)} Saham cocok")
st.caption("Klik satu baris untuk buka halaman detail saham itu.")

event = st.dataframe(
    table_df[final_display_cols].reset_index(drop=True),
    width="stretch",
    hide_index=True,
    height=min(36 * (len(table_df) + 1) + 3, 600),
    column_config=column_config,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    picked_code = table_df.iloc[selected_rows[0]]["stock_code"]
    st.session_state["selected_ticker"] = picked_code
    st.switch_page("pages/1_📈_Detail_Saham.py")
