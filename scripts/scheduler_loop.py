"""Fase 6: simple daily scheduler -- runs scripts.run_daily once per day at
a configured time (default 16:30 WIB, after IDX market close ~16:00 WIB).

Deliberately a plain sleep-until-target loop, not a cron daemon inside the
container (avoids cron/syslog setup for a single job) or an extra scheduling
library (nothing here needs one). `next_run_time()` is a pure function so
the scheduling math can be verified without actually waiting a day.

On an always-on VPS (production, via docker-compose.yml) this loop simply
never misses a day. On a personal machine that isn't running 24/7, two
separate real incidents shaped how this works now:

1. (2026-09-10) A single long `time.sleep()` straight to the next 16:30 --
   on a laptop that suspends, that sleep's deadline is pinned to the wall
   time it was computed at, and Windows pauses the sleep timer while the
   system is suspended, so a machine off across 16:30 WIB had its one
   scheduled run for the day slide silently past the whole day even though
   the process stayed alive the entire time. Fixed by polling every
   SLICE_SECONDS and re-checking the clock + LAST_RUN_MARKER instead of one
   long sleep -- "the slot just arrived" and "we woke up well past it" both
   trip the same run.

2. (2026-09-11) run_is_due() compared LAST_RUN_MARKER as a bare calendar
   DATE ("did a run complete today: yes/no"), blind to WHETHER that run
   happened before or after the market actually closed. A morning catch-up
   run (machine was asleep through a prior slot, woke up before today's
   open, fetched only YESTERDAY's close since today hadn't traded yet) set
   the marker to today's date -- which then satisfied "already ran today"
   for the REST of the day, permanently masking that same evening's real
   post-16:30 run and leaving data one full day stale with the scheduler
   process alive and polling correctly the entire time. Fixed by storing a
   full WIB TIMESTAMP in the marker (see run_daily.py) and comparing it
   against the most recently PASSED 16:30 slot, not the calendar date --
   run_is_due() now covers "never run", "ran before today's slot but it's
   since passed", and "ran days ago, don't wait for the clock" as ONE rule
   instead of the two ad-hoc conditions a previous version special-cased
   (which is exactly how this bug slipped in: catch_up_if_missed() at
   startup and run_is_due() in the poll loop were two separately-maintained
   copies of nearly the same logic that drifted apart). Only one function
   now; catch-up on startup falls out of the poll loop's own first check.
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
# How often the loop re-checks the clock instead of sleeping straight to the
# target. Short enough that a resume-from-suspend past 16:30 fires within a
# few minutes; long enough to be effectively free. Override for tests.
SLICE_SECONDS = int(os.getenv("SCHEDULER_POLL_SECONDS", "600"))


def next_run_time(now: dt.datetime) -> dt.datetime:
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


def last_run_datetime() -> dt.datetime | None:
    """WIB-aware timestamp of the last completed run_daily, or None if it
    never ran / the marker is missing or corrupt. Backward-compatible with
    the OLD bare-date-only marker format (pre-2026-09-11): a bare date
    parses as midnight WIB that day via fromisoformat, which -- being
    earlier than any real run's actual clock time -- can only make a run
    look MORE overdue than it really was, never less; worst case is one
    harmless extra run right after upgrading, never a masked one."""
    try:
        with open(LAST_RUN_MARKER) as f:
            parsed = dt.datetime.fromisoformat(f.read().strip())
    except (OSError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=WIB)


def most_recently_passed_slot(now: dt.datetime) -> dt.datetime:
    """Today's 16:30 WIB if it has already happened, else yesterday's."""
    today_slot = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    return today_slot if now >= today_slot else today_slot - dt.timedelta(days=1)


def run_is_due(now: dt.datetime) -> bool:
    """A run is due whenever the last COMPLETED run happened before the
    most recently passed 16:30 slot. One rule covers every case that used
    to need separate handling: never run before -> due; ran earlier today
    but before 16:30, and 16:30 has since passed -> due (the bug this
    replaced: comparing calendar dates instead of clock time let a morning
    catch-up mask this case all day); ran days ago (machine was off/asleep)
    -> due immediately, not waiting for the clock. Checked on every poll,
    not just at startup, so a resume-from-suspend past the slot gets caught
    within one SLICE_SECONDS tick."""
    last_run = last_run_datetime()
    return last_run is None or last_run < most_recently_passed_slot(now)


def main():
    logger.info(
        "Scheduler started. Target run time: %02d:%02d WIB daily, poll tiap %ds.",
        RUN_HOUR, RUN_MINUTE, SLICE_SECONDS,
    )
    last_logged_target = None
    while True:
        now = dt.datetime.now(WIB)
        if run_is_due(now):
            logger.info("Waktunya jalan (belum ada run setelah slot %02d:%02d WIB terakhir) -- memulai run_daily",
                        RUN_HOUR, RUN_MINUTE)
            try:
                run_daily()
            except Exception:
                logger.exception("run_daily crashed unexpectedly in scheduler loop")
            last_logged_target = None  # force a fresh "next run at" line afterwards
        else:
            target = next_run_time(now)
            if target != last_logged_target:  # log once per target, not every poll
                logger.info("Next run at %s (dalam %.1f jam)", target.isoformat(), (target - now).total_seconds() / 3600)
                last_logged_target = target
        time.sleep(SLICE_SECONDS)


if __name__ == "__main__":
    main()
