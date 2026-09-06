"""Login + approval gate for the whole app.

Call `require_login(page_name)` once, right after `st.set_page_config()`,
on EVERY page (Home.py and every file under app/pages/) -- Streamlit's
multipage apps execute each page script independently, so there is no
single shared entrypoint to hook this into instead; this is the mechanical
but necessary cost of that architecture.

Flow:
1. Not logged in yet -> let them click through GUEST_PREVIEW_PAGES
   distinct pages for free (a taste of the app before being asked to sign
   up, per explicit user request), then show a Google login wall.
2. Logged in, first time ever (no app_users row) -> insert as "pending",
   email the admin, show a "menunggu persetujuan" screen.
3. Logged in, "pending" or "rejected" -> show the matching screen, blocked.
4. Logged in, "approved" -> render the page normally.
5. The admin's own email is auto-approved on first login (bootstrap) --
   otherwise nobody could ever reach the Admin page to approve anyone.
"""
import streamlit as st
from sqlalchemy import select, update

from app.db import app_users, init_schema
from app.email_notify import send_signup_notification
from pipeline.db import get_engine

ADMIN_EMAIL = "heru.purbowiyoto@gmail.com"
GUEST_PREVIEW_PAGES = 2


def _guest_gate(page_name: str) -> bool:
    """True if this page should render normally (still within the free
    preview), False if the login wall should show instead. Tracked in
    st.session_state -- per BROWSER SESSION, not persisted anywhere -- a
    new tab/session gets its own fresh preview. That's fine here: this is
    a UX teaser, not a security boundary (the real gate is the login +
    approval check below, once GUEST_PREVIEW_PAGES is used up)."""
    seen = st.session_state.setdefault("_guest_seen_pages", set())
    if page_name not in seen and len(seen) >= GUEST_PREVIEW_PAGES:
        return False
    seen.add(page_name)
    remaining = GUEST_PREVIEW_PAGES - len(seen)
    if remaining > 0:
        st.info(f"👋 Mode tamu -- {remaining} halaman lagi bisa dilihat gratis sebelum perlu login.", icon="👋")
    else:
        st.info("👋 Mode tamu -- ini halaman gratis terakhir. Halaman berikutnya perlu login.", icon="👋")
    return True


def _show_login_wall() -> None:
    st.title("🔒 Login Diperlukan")
    st.write(
        "Anda sudah melihat pratinjau aplikasi ini. Untuk melanjutkan, silakan login "
        "dengan akun Google -- akses baru aktif setelah disetujui admin."
    )
    if st.button("Login dengan Google", type="primary"):
        try:
            st.login()
        except Exception:
            st.error(
                "Login belum bisa dipakai -- konfigurasi Google OAuth di "
                "`.streamlit/secrets.toml` belum lengkap. Lihat README bagian "
                "'Setup Login' untuk langkahnya."
            )
    st.stop()


def _pending_or_rejected_screen(status: str, email: str) -> None:
    if status == "pending":
        st.title("⏳ Menunggu Persetujuan")
        st.write(f"Akun **{email}** sudah terdaftar dan sedang menunggu persetujuan admin.")
        st.caption("Coba lagi nanti, atau hubungi admin langsung kalau butuh cepat.")
    else:
        st.title("🚫 Akses Ditolak")
        st.write(f"Permintaan akses untuk **{email}** ditolak oleh admin.")
    if st.button("Logout"):
        st.logout()
    st.stop()


def _sidebar_user_badge(email: str) -> None:
    with st.sidebar:
        st.markdown('<div class="mystocks-divider"></div>', unsafe_allow_html=True)
        st.caption(f"Masuk sebagai **{email}**")
        if st.button("Logout", key="_auth_logout_btn"):
            st.logout()


def _upsert_approved(engine, email: str, name: str) -> None:
    with engine.begin() as conn:
        existing = conn.execute(select(app_users).where(app_users.c.email == email)).fetchone()
        if existing is None:
            conn.execute(app_users.insert().values(email=email, name=name, status="approved"))
        elif existing.status != "approved":
            conn.execute(update(app_users).where(app_users.c.email == email).values(status="approved"))


def require_login(page_name: str, allow_guest_preview: bool = True) -> None:
    is_logged_in = getattr(st.user, "is_logged_in", False)

    if not is_logged_in:
        # Admin (and any other sensitive page that opts out) skips the
        # guest-preview allowance entirely -- letting a not-yet-logged-in
        # visitor reach it via their free preview pages would hit
        # `st.user.email` below with no such attribute set at all.
        if allow_guest_preview and _guest_gate(page_name):
            return
        _show_login_wall()
        return  # unreachable (_show_login_wall calls st.stop()), kept for clarity

    email = st.user.email
    name = getattr(st.user, "name", None) or email

    engine = get_engine()
    init_schema(engine)

    if email == ADMIN_EMAIL:
        _upsert_approved(engine, email, name)
        _sidebar_user_badge(email)
        return

    with engine.connect() as conn:
        row = conn.execute(select(app_users).where(app_users.c.email == email)).fetchone()

    if row is None:
        with engine.begin() as conn:
            conn.execute(app_users.insert().values(email=email, name=name, status="pending"))
        try:
            send_signup_notification(email, name)
        except Exception:
            pass  # notification failing must never block the sign-up itself
        _pending_or_rejected_screen("pending", email)
        return

    if row.status == "approved":
        _sidebar_user_badge(email)
        return

    _pending_or_rejected_screen(row.status, email)
