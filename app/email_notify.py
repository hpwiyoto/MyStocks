"""Sends the admin an email when a new Google account signs in to MyStocks
for the first time (lands in "pending" status). Uses Gmail's SMTP server
with an App Password (NOT the Gmail account password itself -- Google
blocks plain-password SMTP login; see README's "Setup Login" section for
how to generate one). Credentials live in `.streamlit/secrets.toml`
alongside the OIDC auth config, since both are the same trust boundary
(Streamlit's own secrets store) -- not a new env-var scheme just for this.

Callers (app/auth.py) always wrap this in a try/except: a failed
notification must never block someone's ability to sign in and land in
the "pending" queue -- it would just mean the admin has to check the
Admin page manually instead of getting pinged.
"""
import smtplib
from email.mime.text import MIMEText

import streamlit as st


def send_signup_notification(applicant_email: str, applicant_name: str) -> None:
    cfg = st.secrets["gmail"]
    sender = cfg["address"]
    app_password = cfg["app_password"]
    recipient = cfg.get("notify_to", sender)

    body = (
        "Ada permintaan akses baru ke MyStocks.\n\n"
        f"Nama : {applicant_name}\n"
        f"Email: {applicant_email}\n\n"
        "Buka halaman Admin di aplikasi (menu sidebar) untuk approve/reject."
    )
    msg = MIMEText(body)
    msg["Subject"] = f"[MyStocks] Permintaan akses baru: {applicant_email}"
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, [recipient], msg.as_string())
