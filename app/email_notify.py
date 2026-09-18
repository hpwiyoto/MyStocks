"""Email notifications for MyStocks -- new-signup alerts (app/auth.py) and
new-BUY-signal alerts (scripts/notify_buy_signals.py, wired into
scripts/run_daily.py). Uses Gmail's SMTP server with an App Password (NOT
the Gmail account password itself -- Google blocks plain-password SMTP
login; see README's "Setup Login" section for how to generate one).
Credentials live in `.streamlit/secrets.toml` alongside the OIDC auth
config, since both are the same trust boundary (Streamlit's own secrets
store) -- not a new env-var scheme just for this. `st.secrets` reads that
file directly and works fine outside a running Streamlit server too (e.g.
from the plain scripts.run_daily batch process), no app context needed.

Every caller wraps these in a try/except: a failed notification must
never block the thing that triggered it (someone signing in, or the
daily pipeline finishing) -- it would just mean the admin has to check
the app manually instead of getting pinged.
"""
import smtplib
from email.mime.text import MIMEText

import streamlit as st


def _approved_user_emails() -> list[str]:
    """Every email currently allowed to log into MyStocks (app_users
    status='approved') -- direct user request for BUY-signal alerts to
    go to "the address that logs into this app" rather than one fixed
    address in secrets.toml, so it stays correct if the admin approves
    more people later without anyone needing to touch secrets.toml.
    Empty list if the table doesn't exist yet or has no approved rows
    (e.g. scripts.run_daily's very first-ever run, before anyone has
    logged in once) -- callers fall back to the sender's own address.
    """
    from sqlalchemy import select

    from app.db import app_users, init_schema
    from pipeline.db import get_engine

    engine = get_engine()
    init_schema(engine)
    with engine.connect() as conn:
        rows = conn.execute(select(app_users.c.email).where(app_users.c.status == "approved")).fetchall()
    return [r[0] for r in rows]


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


def send_buy_signal_notification(new_buys: list[dict]) -> None:
    """One daily email listing every ticker that just turned BUY today
    for ANY of the 5 Swing configs -- direct user request. `new_buys`:
    list of {"config_label", "stock_code", "probability", "entry_price",
    "take_profit_price", "stop_loss_price"} from
    scripts.notify_buy_signals.find_new_buy_signals -- already filtered
    to NEWLY-appeared BUYs only (per user's explicit choice: not a daily
    repeat of every still-BUY ticker, only the ones that weren't BUY
    yesterday, to avoid spamming the same signal every day it stays BUY)
    and to tradeable tickers (suspended/ARA/Special Monitoring Board
    already excluded by the caller). No-ops on an empty list -- callers
    should check `if new_buys:` themselves if they want to skip the
    SMTP round-trip entirely, but this is harmless to call regardless.

    Sent to every currently-approved app user (_approved_user_emails) --
    direct user request ("kirim emailnya ke alamat yang login ke
    aplikasi ini"), NOT the fixed notify_to/sender address
    send_signup_notification uses (that one stays admin-only on purpose:
    approving signups is an admin action, but a BUY signal is relevant
    to everyone using the app). Falls back to notify_to/sender if no one
    is approved yet.

    Same "caller wraps in try/except" contract as send_signup_notification
    -- see this module's docstring.
    """
    if not new_buys:
        return
    cfg = st.secrets["gmail"]
    sender = cfg["address"]
    app_password = cfg["app_password"]
    recipients = _approved_user_emails() or [cfg.get("notify_to", sender)]

    by_config: dict[str, list[dict]] = {}
    for row in new_buys:
        by_config.setdefault(row["config_label"], []).append(row)

    lines = [f"{len(new_buys)} saham baru BUY hari ini di Swing MyStocks.\n"]
    for label, rows in by_config.items():
        lines.append(f"\n{label} ({len(rows)} saham):")
        for r in rows:
            lines.append(
                f"  - {r['stock_code']}: probabilitas {r['probability'] * 100:.1f}%, "
                f"entry {r['entry_price']:,.0f}, target {r['take_profit_price']:,.0f}, "
                f"stop {r['stop_loss_price']:,.0f}"
            )
    lines.append("\nBuka halaman Swing di aplikasi untuk detail lengkap dan menandai posisi.")
    body = "\n".join(lines)
    subject = f"[MyStocks] {len(new_buys)} sinyal BUY baru hari ini"

    # One SMTP login, one message per recipient (own To: header) rather
    # than one message with everyone in To: -- approved users don't need
    # to see each other's email addresses.
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
        server.starttls()
        server.login(sender, app_password)
        for recipient in recipients:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = recipient
            server.sendmail(sender, [recipient], msg.as_string())
