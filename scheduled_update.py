"""Cron entry point for the data update.

Cron calls this every five minutes; it exits in milliseconds unless an update is
actually due, so the schedule lives here rather than in the crontab:

  - once a day at DAILY_TIME
  - on every invocation inside a Nobel announcement window, i.e. from the
    announced time until WINDOW_HOURS later on the days in ANNOUNCEMENTS

All times are Europe/Stockholm, where the prizes are announced. That is the
reason this file exists at all: the server clock runs on UTC, so putting 11:30
into the crontab would fire two hours late during CEST -- and one hour off once
winter time starts. Converting here keeps the table readable and correct.

Only the standard library is imported at module level. update_data.py (pandas,
polars, the full API fetch) runs as a subprocess, and only when there is work.

Usage:
    scheduled_update.py            # what cron runs
    scheduled_update.py --check    # print the decision, change nothing
    scheduled_update.py --force    # run the update regardless of schedule
"""

import fcntl
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TZ = ZoneInfo("Europe/Stockholm")

# Earliest announced time per prize, in Stockholm local time. Refresh once a
# year from https://www.nobelprize.org/press/ -- the dates shift, the pattern
# (Medicine Monday through Peace Friday, Economics the following Monday) holds.
ANNOUNCEMENTS = {
    "2026-10-05": ("11:30", "Medicine"),
    "2026-10-06": ("11:45", "Physics"),
    "2026-10-07": ("11:45", "Chemistry"),
    "2026-10-08": ("13:00", "Literature"),
    "2026-10-09": ("11:00", "Peace"),
    "2026-10-12": ("11:45", "Economics"),
}

# Announcements regularly start late and the API needs a moment to follow, so
# the window stays open well past the scheduled time.
WINDOW_HOURS = 3

# Baseline update for the rest of the year.
DAILY_TIME = "08:00"

STATE_FILE = os.path.join(BASE_DIR, "last_scheduled_run.txt")
LOCK_FILE = os.path.join(BASE_DIR, ".scheduled_update.lock")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'data_update.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('scheduled_update')


def announcement_window(now):
    """Return the prize name if `now` sits inside an announcement window."""
    entry = ANNOUNCEMENTS.get(now.strftime("%Y-%m-%d"))
    if entry is None:
        return None

    start_time, prize = entry
    hour, minute = (int(part) for part in start_time.split(":"))
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if start <= now < start + timedelta(hours=WINDOW_HOURS):
        return prize
    return None


def read_last_run():
    """Date (YYYY-MM-DD) of the last successful run, or None."""
    try:
        with open(STATE_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning(f"Could not read {STATE_FILE}: {e}")
        return None


def write_last_run(day):
    try:
        with open(STATE_FILE, 'w') as f:
            f.write(day)
    except OSError as e:
        logger.error(f"Could not write {STATE_FILE}: {e}")


def daily_due(now):
    """True once DAILY_TIME has passed and today's update has not run yet."""
    hour, minute = (int(part) for part in DAILY_TIME.split(":"))
    if (now.hour, now.minute) < (hour, minute):
        return False
    return read_last_run() != now.strftime("%Y-%m-%d")


def decide(now):
    """Return the reason to run, or None to stay idle."""
    prize = announcement_window(now)
    if prize:
        return f"{prize} announcement window"
    if daily_due(now):
        return "daily update"
    return None


def run_update():
    """Run update_data.py in this interpreter's environment."""
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "update_data.py")],
        cwd=BASE_DIR
    )
    return result.returncode


def main():
    args = sys.argv[1:]
    now = datetime.now(TZ)
    reason = "manual --force" if "--force" in args else decide(now)

    if "--check" in args:
        stamp = now.strftime('%Y-%m-%d %H:%M %Z')
        print(f"Now (Stockholm): {stamp}")
        print(f"Last run:        {read_last_run() or 'never'}")
        print(f"Decision:        {reason or 'idle - nothing due'}")
        return 0

    if reason is None:
        return 0

    # A slow API run can outlast the five-minute cron interval; without the lock
    # two updates would write the same CSVs at the same time.
    with open(LOCK_FILE, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("Another update is still running - skipping this slot")
            return 0

        logger.info(f"Triggering update: {reason}")
        returncode = run_update()

    if returncode == 0:
        write_last_run(now.strftime("%Y-%m-%d"))
    else:
        logger.error(f"update_data.py exited with {returncode}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
