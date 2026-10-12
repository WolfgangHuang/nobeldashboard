"""
Locations of all data files. Standard library only, so the pipeline scripts can
import it without pulling in pandas/Dash.

    data/raw/      fetched or scraped: df_laureates.csv from the API (not in git,
                   written by the update cron), nominations_full.csv from the scraper
    data/curated/  maintained by hand (corrections, ethnicity, religion, ...)
    data/derived/  exports written by pipeline/export_derived.py (not in git)
    data/state/    runtime state: timestamps, lock, log (not in git)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
DERIVED_DIR = DATA_DIR / "derived"
STATE_DIR = DATA_DIR / "state"


def raw(name):
    return RAW_DIR / name


def curated(name):
    return CURATED_DIR / name


def derived(name):
    return DERIVED_DIR / name


def state(name):
    return STATE_DIR / name
