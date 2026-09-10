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
either today's slot has already passed with no run today, OR the last
successful run is more than one day behind regardless of the clock (a
Codespace idle for 9 days and restarted mid-morning, well before today's
slot, is a real case the first condition alone missed entirely --
confirmed live, it just sat waiting several more hours for data that was
already over a week stale). run_daily() is fully idempotent (see its own
docstring), so this is safe even if it ends up racing a normal run.

The main loop no longer does one long `time.sleep()` straight to the next
16:30 -- on a laptop that suspends, that single sleep's deadline is pinned
to the wall time it was computed at, and Windows pauses the sleep timer
while the system is suspended, so a machine off across 16:30 WIB has its
one scheduled run for the day slide silently past the whole day even
though the process stays alive the entire time (confirmed real: 09-10-2026,
PID alive since the day before, zero run_daily activity that day, data
frozen at 09-09). Instead it polls every SLICE_SECONDS and re-checks the
clock + LAST_RUN_MARKER, so "the slot just arrived" and "we woke up well
past it" both trip the same catch-up, and last_run_date() keeps it to one
run per calendar day.
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


def last_run_date() -> dt.date | None:
    try:
        with open(LAST_RUN_MARKER) as f:
            return dt.date.fromisoformat(f.read().strip())
    except (OSError, ValueError):
        return None  # never run yet, or marker missing/corrupt -- treat as "stale"


def catch_up_if_missed(now: dt.datetime) -> None:
    today = now.date()
    last_run = last_run_date()
    if last_run == today:
        return  # already ran today (this is just an ordinary process restart)

    # Two independent reasons to catch up right now instead of waiting for
    # the loop below:
    # 1. Today's slot has already passed with no successful run today -- the
    #    original check. Handles "machine was off exactly through today's
    #    16:30 WIB".
    # 2. last_run is MORE than one day behind, regardless of whether today's
    #    slot has arrived yet. Before today's slot, last_run == yesterday (or
    #    the last trading day) is the NORMAL, expected state -- that run
    #    simply hasn't happened yet today. But if it's further behind than
    #    that (confirmed via a real incident: a Codespace idle for 9 days,
    #    restarted mid-morning well before 16:30 WIB -- reason 1 alone left
    #    this scenario waiting several more hours for data that was already
    #    over a week stale), there's no reason to wait for the clock at all.
    target_today = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    missed_todays_slot = now >= target_today
    missed_a_prior_day = last_run is None or (today - last_run).days > 1
    if not (missed_todays_slot or missed_a_prior_day):
        return  # normal pre-slot state, the loop below will handle it on time

    logger.info(
        "Belum ada run sukses hari ini (terakhir: %s, target slot %02d:%02d WIB) -- "
        "kemungkinan mesin ini mati/tidur saat satu atau lebih jadwal terlewat. "
        "Menjalankan catch-up run_daily sekarang, tidak menunggu slot berikutnya.",
        last_run.isoformat() if last_run else "belum pernah", RUN_HOUR, RUN_MINUTE,
    )
    try:
        run_daily()
    except Exception:
        logger.exception("Catch-up run_daily crashed unexpectedly")


def run_is_due(now: dt.datetime) -> bool:
    """Should run_daily fire right now? True when no successful run has
    completed on today's date AND either today's 16:30 slot has arrived, or
    the last run is more than a full day stale (don't wait for the clock to
    catch up a machine that was suspended for days). Same two conditions as
    catch_up_if_missed, evaluated every poll rather than only at startup --
    that's what makes a resume-from-suspend past the slot get caught."""
    last_run = last_run_date()
    if last_run == now.date():
        return False
    target_today = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    missed_todays_slot = now >= target_today
    missed_a_prior_day = last_run is None or (now.date() - last_run).days > 1
    return missed_todays_slot or missed_a_prior_day


def main():
    logger.info(
        "Scheduler started. Target run time: %02d:%02d WIB daily, poll tiap %ds.",
        RUN_HOUR, RUN_MINUTE, SLICE_SECONDS,
    )
    catch_up_if_missed(dt.datetime.now(WIB))
    last_logged_target = None
    while True:
        now = dt.datetime.now(WIB)
        if run_is_due(now):
            logger.info("Waktunya jalan (belum ada run sukses %s) -- memulai run_daily", now.date().isoformat())
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
