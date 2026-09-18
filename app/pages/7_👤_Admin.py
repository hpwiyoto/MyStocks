import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from sqlalchemy import delete, select, update

from app.auth import ADMIN_EMAIL, require_login
from app.data import load_manual_exclusion_tickers
from app.db import app_users, init_schema, manual_ticker_exclusion
from app.style import inject_base_css, render_developer_footer
from pipeline.db import get_engine, upsert

st.set_page_config(page_title="MyStocks — Admin", page_icon="👤", layout="wide")
inject_base_css()
# allow_guest_preview=False: this page is sensitive enough that a
# not-yet-logged-in visitor should never reach it via their free preview
# pages -- require_login() would otherwise let them through, and the
# st.user.email check right below has no such attribute at all pre-login.
require_login("Admin", allow_guest_preview=False)
render_developer_footer()

if st.button("← Kembali ke Home"):
    st.switch_page("Home.py")

# require_login() already guarantees the visitor is logged in AND
# approved by this point -- this second check is specifically about
# WHICH approved user they are, since "approved" alone doesn't mean
# "admin" (every non-admin approved user also passes require_login fine).
if getattr(st.user, "email", None) != ADMIN_EMAIL:
    st.error("🚫 Halaman ini hanya untuk admin.")
    st.stop()

st.title("👤 Admin — Persetujuan Akses")
st.caption(
    "Menyetujui atau menolak permintaan akses ke MyStocks. Setiap akun Google yang login "
    "pertama kali masuk sebagai 'Menunggu' di sini sampai diputuskan di halaman ini."
)

engine = get_engine()
init_schema(engine)

with engine.connect() as conn:
    rows = conn.execute(select(app_users).order_by(app_users.c.requested_at.desc())).fetchall()

pending = [r for r in rows if r.status == "pending"]
approved = [r for r in rows if r.status == "approved"]
rejected = [r for r in rows if r.status == "rejected"]

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.subheader(f"⏳ Menunggu persetujuan ({len(pending)})")
if not pending:
    st.caption("Tidak ada permintaan yang menunggu.")
for r in pending:
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.write(f"**{r.name}**  \n{r.email}")
    if c2.button("✅ Approve", key=f"approve_{r.email}", width="stretch"):
        with engine.begin() as conn:
            conn.execute(update(app_users).where(app_users.c.email == r.email).values(status="approved"))
        st.rerun()
    if c3.button("🚫 Reject", key=f"reject_{r.email}", width="stretch"):
        with engine.begin() as conn:
            conn.execute(update(app_users).where(app_users.c.email == r.email).values(status="rejected"))
        st.rerun()

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.subheader(f"✅ Disetujui ({len(approved)})")
if approved:
    st.dataframe(
        [{"Nama": r.name, "Email": r.email, "Sejak": r.requested_at} for r in approved],
        hide_index=True, width="stretch",
    )
else:
    st.caption("Belum ada yang disetujui.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.subheader(f"🚫 Ditolak ({len(rejected)})")
if rejected:
    dt1, dt2 = st.columns([3, 1])
    for r in rejected:
        dt1.write(f"{r.name} -- {r.email}")
        if dt2.button("↩️ Approve sekarang", key=f"undo_reject_{r.email}"):
            with engine.begin() as conn:
                conn.execute(update(app_users).where(app_users.c.email == r.email).values(status="approved"))
            st.rerun()
else:
    st.caption("Belum ada yang ditolak.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
manual_df = load_manual_exclusion_tickers()
st.subheader(f"🚫 Saham Dikecualikan dari Screener ({len(manual_df)})")
st.caption(
    "Saham yang tidak bisa ditransaksikan normal -- FCA/Papan Pemantauan Khusus BEI, atau alasan "
    "manual lain -- dan karena itu dikecualikan dari SEMUA screener (Home, Swing, Momentum, "
    "Rekomendasi Emitten, Screener Kustom) begitu disimpan di sini, TANPA perlu restart aplikasi "
    "(lihat `app.data.load_suspended_tickers`/`untradeable_reason`). Riwayat lengkap kapan tiap "
    "saham masuk/keluar Papan Pemantauan Khusus ada di `scripts/special_monitoring_board.py`, tapi "
    "aplikasi ini HANYA membaca tabel di bawah -- update di sini, bukan di file itu, supaya "
    "perubahan langsung berlaku bagi semua pengguna."
)

if not manual_df.empty:
    st.dataframe(
        manual_df, hide_index=True, width="stretch",
        height=min(36 * (len(manual_df) + 1) + 3, 400),
        column_config={
            "stock_code": st.column_config.TextColumn("Kode"),
            "reason": st.column_config.TextColumn("Alasan"),
            "note": st.column_config.TextColumn("Catatan"),
            "updated_at": st.column_config.DatetimeColumn("Diperbarui"),
        },
    )
else:
    st.caption("Belum ada data.")

st.markdown("##### 🔄 Timpa Daftar (overwrite berdasarkan alasan)")
st.caption(
    "Tempel daftar kode saham lengkap (mis. dari PDF BEI terbaru) -- SEMUA baris dengan alasan yang "
    "sama akan DIHAPUS lalu diganti dengan daftar baru ini. Ini SENGAJA overwrite, bukan tambah, "
    "supaya tabel ini selalu jadi snapshot saat ini saja (hemat memori) bukan riwayat yang terus "
    "menumpuk."
)
with st.form("admin_overwrite_form"):
    ov_reason = st.text_input("Alasan", value="special_monitoring")
    ov_codes = st.text_area(
        "Daftar kode saham (pisahkan spasi, koma, atau baris baru)", height=150,
        placeholder="TGUK, SAFE, TRUK\nPACK ...",
    )
    ov_note = st.text_input("Catatan (opsional)", placeholder="mis. Update PDF BEI 18 Sep 2026")
    ov_confirm_empty = st.checkbox("Daftar di atas SENGAJA saya kosongkan -- hapus semua entri alasan ini")
    ov_submit = st.form_submit_button("🔄 Timpa & Simpan")

if ov_submit:
    codes = sorted({c for c in re.split(r"[\s,]+", ov_codes.strip().upper()) if c})
    if not codes and not ov_confirm_empty:
        st.error(
            "Daftar kosong -- ini akan menghapus SEMUA entri untuk alasan ini. Centang kotak "
            "konfirmasi di atas kalau itu memang maksud Anda, lalu kirim ulang."
        )
    else:
        engine = get_engine()
        init_schema(engine)
        with engine.begin() as conn:
            conn.execute(delete(manual_ticker_exclusion).where(manual_ticker_exclusion.c.reason == ov_reason))
            if codes:
                rows = [{"stock_code": c, "reason": ov_reason, "note": ov_note or None} for c in codes]
                upsert(conn, manual_ticker_exclusion, rows, update_columns=["reason", "note"], index_elements=["stock_code"])
        load_manual_exclusion_tickers.clear()
        st.success(f"Berhasil menimpa {len(codes)} kode untuk alasan '{ov_reason}'.")
        st.rerun()

st.markdown("##### ➕ Tambah / Hapus Satu Ticker")
st.caption("Untuk perubahan kecil (satu saham keluar/masuk papan) tanpa menimpa seluruh daftar.")
add1, add2, add3, add4 = st.columns([1.2, 1.2, 2, 1])
add_code = add1.text_input("Kode", key="admin_add_code", placeholder="mis. TGUK")
add_reason = add2.text_input("Alasan", key="admin_add_reason", value="special_monitoring")
add_note = add3.text_input("Catatan", key="admin_add_note", placeholder="opsional")
with add4:
    st.markdown("<div style='height:1.6em'></div>", unsafe_allow_html=True)
    if st.button("+ Tambah", key="admin_add_btn", width="stretch"):
        code = add_code.strip().upper()
        if code:
            engine = get_engine()
            init_schema(engine)
            with engine.begin() as conn:
                upsert(
                    conn, manual_ticker_exclusion,
                    [{"stock_code": code, "reason": add_reason.strip() or "special_monitoring", "note": add_note.strip() or None}],
                    update_columns=["reason", "note"], index_elements=["stock_code"],
                )
            load_manual_exclusion_tickers.clear()
            st.success(f"{code} ditambahkan/diperbarui.")
            st.rerun()

if not manual_df.empty:
    rm1, rm2 = st.columns([3, 1])
    remove_code = rm1.selectbox("Hapus ticker", manual_df["stock_code"].tolist(), key="admin_remove_select")
    with rm2:
        st.markdown("<div style='height:1.6em'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Hapus", key="admin_remove_btn", width="stretch"):
            engine = get_engine()
            with engine.begin() as conn:
                conn.execute(delete(manual_ticker_exclusion).where(manual_ticker_exclusion.c.stock_code == remove_code))
            load_manual_exclusion_tickers.clear()
            st.success(f"{remove_code} dihapus.")
            st.rerun()
