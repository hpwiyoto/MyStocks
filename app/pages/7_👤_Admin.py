import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from sqlalchemy import select, update

from app.auth import ADMIN_EMAIL, require_login
from app.db import app_users, init_schema
from app.style import inject_base_css, render_developer_footer
from pipeline.db import get_engine

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
