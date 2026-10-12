"""Shared logging setup for the pipeline scripts (standard library only)."""

import logging
from logging.handlers import RotatingFileHandler

from nbldash import paths

LOG_FILE = paths.state("data_update.log")


def setup_logging(name):
    """
    Log to data/state/data_update.log and stderr.

    The file rotates at 1 MB and keeps three old copies: during the announcement
    windows the update runs every five minutes, and the log used to grow forever.
    """
    paths.STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)
