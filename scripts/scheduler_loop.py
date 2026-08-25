"""Fase 6: simple daily scheduler -- runs scripts.run_daily once per day at
a configured time (default 16:30 WIB, after IDX market close ~16:00 WIB).

Deliberately a plain sleep-until-target loop, not a cron daemon inside the
container (avoids cron/syslog setup for a single job) or an extra scheduling
library (nothing here needs one). `next_run_time()` is a pure function so
the scheduling math can be verified without actually waiting a day.
"""
import datetime as dt
import os
import time
import zoneinfo

from pipeline.logging_config import get_logger
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


def main():
    logger.info("Scheduler started. Target run time: %02d:%02d WIB daily.", RUN_HOUR, RUN_MINUTE)
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
