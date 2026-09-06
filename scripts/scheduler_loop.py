"""Fase 6: simple daily scheduler -- runs scripts.run_daily once per day at
a configured time (default 16:30 WIB, after IDX market close ~16:00 WIB).

Deliberately a plain sleep-until-target loop, not a cron daemon inside the
container (avoids cron/syslog setup for a single job) or an extra scheduling
library (nothing here needs one). `next_run_time()` is a pure function so
the scheduling math can be verified without actually waiting a day.

On an always-on VPS (production, via docker-compose.yml) this loop simply
never misses a day. On a personal machine that isn't running 24/7 (e.g. a
laptop closed/off overnight or through a weekend -- confirmed via a real
user report of stale prices/indicators, root-caused to exactly this: the
scheduler process wasn't alive at 16:30 WIB on 4 straight days, and each
restart afterwards just rescheduled to TOMORROW instead of catching up),
the plain "wait for the next clock time" loop silently leaves data stale
for however long the machine was off. `main()` now checks
run_daily.LAST_RUN_MARKER on startup and runs an immediate catch-up if
today's slot has already passed and no run has completed yet today --
run_daily() is fully idempotent (see its own docstring), so this is safe
even if it ends up racing a normal run.
"""
import datetime as dt
import os
import time
import zoneinfo

from pipeline.logging_config import get_logger
from scripts.run_daily import LAST_RUN_MARKER
from scripts.run_daily import run as run_daily

logger = get_logger("scripts.scheduler_loop")

WIB = zoneinfo.ZoneInfo("Asia/Jakarta")
RUN_HOUR = int(os.getenv("SCHEDULER_RUN_HOUR", "16"))
RUN_MINUTE = int(os.getenv("SCHEDULER_RUN_MINUTE", "30"))


def next_run_time(now: dt.datetime) -> dt.datetime:
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


def last_run_date() -> dt.date | None:
    try:
        with open(LAST_RUN_MARKER) as f:
            return dt.date.fromisoformat(f.read().strip())
    except (OSError, ValueError):
        return None  # never run yet, or marker missing/corrupt -- treat as "stale"


def catch_up_if_missed(now: dt.datetime) -> None:
    target_today = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now < target_today:
        return  # today's slot hasn't arrived yet -- the normal loop below will handle it
    last_run = last_run_date()
    if last_run == now.date():
        return  # already ran today (this is just an ordinary process restart)
    logger.info(
        "Slot %02d:%02d WIB hari ini sudah lewat dan belum ada run sukses hari ini "
        "(terakhir: %s) -- kemungkinan mesin ini mati/tidur saat jadwal terlewat. "
        "Menjalankan catch-up run_daily sekarang.",
        RUN_HOUR, RUN_MINUTE, last_run.isoformat() if last_run else "belum pernah",
    )
    try:
        run_daily()
    except Exception:
        logger.exception("Catch-up run_daily crashed unexpectedly")


def main():
    logger.info("Scheduler started. Target run time: %02d:%02d WIB daily.", RUN_HOUR, RUN_MINUTE)
    catch_up_if_missed(dt.datetime.now(WIB))
    while True:
        now = dt.datetime.now(WIB)
        target = next_run_time(now)
        sleep_seconds = (target - now).total_seconds()
        logger.info("Next run at %s (dalam %.1f jam)", target.isoformat(), sleep_seconds / 3600)
        time.sleep(sleep_seconds)
        logger.info("Waktunya jalan -- memulai run_daily")
        try:
            run_daily()
        except Exception:
            logger.exception("run_daily crashed unexpectedly in scheduler loop")


if __name__ == "__main__":
    main()
