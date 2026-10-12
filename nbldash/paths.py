"""
Locations of all data files. Standard library only, so the pipeline scripts can
import it without pulling in pandas/Dash.

In git (part of the code, baked into the image):
    data/curated/                maintained by hand (corrections, ethnicity, religion, ...)
    data/raw/nominations_full.csv  from the scraper (pipeline/scrape_nominations.py)

Written at runtime by the update cron (not in git), under DATA_DIR:
    raw/df_laureates.csv         from the Nobel Prize API
    derived/                     exports written by pipeline/export_derived.py
    state/                       timestamps, lock, log

DATA_DIR is data/ in the checkout unless NBL_DATA_DIR points elsewhere. Under
Dokploy it must: the checkout is re-cloned on every deploy, so the runtime data
lives in the app's files/ directory instead (see README, Deployment).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_DATA_DIR = ROOT / "data"
CURATED_DIR = REPO_DATA_DIR / "curated"
NOMINATIONS_CSV = REPO_DATA_DIR / "raw" / "nominations_full.csv"

DATA_DIR = Path(os.environ.get("NBL_DATA_DIR") or REPO_DATA_DIR).resolve()
RAW_DIR = DATA_DIR / "raw"
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
