import datetime as dt
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
from scripts.parse_special_monitoring_pdf import active_tickers, parse_pdf_bytes

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

st.markdown("##### 📄 Upload PDF Otomatis (disarankan)")
st.caption(
    "Upload langsung PDF resmi 'Papan Pemantauan Khusus' dari BEI -- diparse otomatis (kode saham, "
    "tanggal masuk, tanggal keluar), ditampilkan sebagai pratinjau + perbedaan dari daftar saat ini, "
    "baru diterapkan setelah Anda klik konfirmasi. Tidak ada yang berubah di database sampai Anda "
    "menekan tombol 'Terapkan' di bawah."
)
pdf_reason = st.text_input("Alasan untuk hasil upload ini", value="special_monitoring", key="pdf_upload_reason")
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

if uploaded_pdf is not None:
    try:
        pdf_rows, pdf_skipped = parse_pdf_bytes(uploaded_pdf.getvalue())
    except Exception as exc:
        st.error(f"Gagal membaca PDF ini: {exc}")
        pdf_rows, pdf_skipped = [], []

    if pdf_rows:
        new_active = active_tickers(pdf_rows)
        current_active = set(manual_df.loc[manual_df["reason"] == pdf_reason, "stock_code"]) if not manual_df.empty else set()
        added = sorted(new_active - current_active)
        removed = sorted(current_active - new_active)
        unchanged = sorted(new_active & current_active)

        st.info(
            f"📄 {len(pdf_rows)} baris berhasil diparse, **{len(new_active)} saham sedang aktif** "
            "(belum ada tanggal keluar) menurut PDF ini."
        )
        if pdf_skipped:
            with st.expander(f"⚠️ {len(pdf_skipped)} baris terlihat seperti data tapi gagal diparse -- klik untuk cek"):
                for s in pdf_skipped:
                    st.text(s)

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("➕ Akan ditambahkan", len(added))
        pc2.metric("➖ Akan dihapus (sudah keluar)", len(removed))
        pc3.metric("= Tidak berubah", len(unchanged))
        if added:
            st.caption("➕ " + ", ".join(added))
        if removed:
            st.caption("➖ " + ", ".join(removed))

        if st.button("✅ Terapkan Hasil Parsing PDF", key="pdf_apply_btn", type="primary"):
            engine = get_engine()
            init_schema(engine)
            with engine.begin() as conn:
                conn.execute(delete(manual_ticker_exclusion).where(manual_ticker_exclusion.c.reason == pdf_reason))
                if new_active:
                    note = f"Auto-parse PDF BEI ({dt.date.today().isoformat()})"
                    rows_to_upsert = [{"stock_code": c, "reason": pdf_reason, "note": note} for c in sorted(new_active)]
                    upsert(conn, manual_ticker_exclusion, rows_to_upsert, update_columns=["reason", "note"], index_elements=["stock_code"])
            load_manual_exclusion_tickers.clear()
            st.success(f"Berhasil menerapkan {len(new_active)} saham aktif dari PDF untuk alasan '{pdf_reason}'.")
            st.rerun()
    elif pdf_skipped:
        st.warning(
            f"Tidak ada baris valid yang berhasil diparse ({len(pdf_skipped)} baris diabaikan). "
            "Cek isi PDF-nya di atas -- mungkin bukan PDF Papan Pemantauan Khusus, atau formatnya "
            "beda dari yang diharapkan."
        )
    else:
        st.warning("Tidak ada data yang bisa diparse dari PDF ini -- pastikan ini PDF 'Papan Pemantauan Khusus' resmi BEI.")

st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
st.markdown("##### ✍️ Atau: Tempel Manual (fallback kalau upload PDF gagal / format berubah)")
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
