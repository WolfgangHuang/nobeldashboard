"""
Nobel Prize Data Update Script
Standalone script to update data from Nobel Prize API
Designed to be run via cron job or systemd timer, from the project root:

    python -m pipeline.update_data [--force]

Processing and the app restart only happen when the API data differs from the
data the last complete run processed; --force does both regardless.
"""

import hashlib
import shutil
import sys
import os
import subprocess
import pandas as pd
from datetime import datetime, timezone

from nbldash import paths
from pipeline.logs import setup_logging
from pipeline.nobel_api import get_all_laureates_data, save_laureates_data

ROOT = str(paths.ROOT)
LAUREATES_CSV = paths.raw("df_laureates.csv")
# The previous file, kept while the new one is being checked by the export step.
LAUREATES_BACKUP = paths.state("df_laureates.previous.csv")

logger = setup_logging('data_update')

# Digest of the df_laureates.csv the last *complete* run (processing + restart)
# was based on. Written only after both succeeded, so a run that failed halfway is
# retried next time even though the raw file is already up to date.
PROCESSED_STATE = paths.state("last_processed_laureates.sha256")


def file_digest(path):
    """SHA-256 of a file's bytes, or None if it does not exist."""
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def read_processed_digest():
    try:
        with open(PROCESSED_STATE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def write_processed_digest(digest):
    try:
        with open(PROCESSED_STATE, 'w') as f:
            f.write(digest)
    except OSError as e:
        logger.error(f"Could not write {PROCESSED_STATE}: {e}")


def restore_previous_data():
    """Put the df_laureates.csv from before this run back in place."""
    if os.path.exists(LAUREATES_BACKUP):
        shutil.copy2(LAUREATES_BACKUP, LAUREATES_CSV)
        logger.warning(f"Restored the previous {LAUREATES_CSV.name}")


def save_update_timestamp():
    """Save the current timestamp to file."""
    timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    try:
        with open(paths.state("last_api_update.txt"), 'w') as f:
            f.write(timestamp)
        logger.info(f"Timestamp saved: {timestamp}")
    except Exception as e:
        logger.error(f"Failed to save timestamp: {e}")


def validate_laureates_data(df_new, df_existing=None):
    """Validate that the new data is complete and reasonable."""
    if df_new is None or df_new.empty:
        return False, "DataFrame is None or empty"

    # Check minimum expected record count
    if len(df_new) < 900:
        return False, f"Too few records: {len(df_new)} (expected 900+)"

    # Check for essential columns
    essential_columns = ['ID_Laureate', 'AwardeeDisplayName', 'Prize0_AwardYear', 'Prize0_Category']
    missing_columns = [col for col in essential_columns if col not in df_new.columns]
    if missing_columns:
        return False, f"Missing essential columns: {missing_columns}"

    # Check for reasonable number of null values in key fields
    null_awardee_names = df_new['AwardeeDisplayName'].isnull().sum()
    if null_awardee_names > len(df_new) * 0.1:
        return False, f"Too many null awardee names: {null_awardee_names}"

    # If we have existing data, compare sizes
    if df_existing is not None and not df_existing.empty:
        if len(df_new) < len(df_existing) * 0.95:
            return False, f"New data significantly smaller than existing: {len(df_new)} vs {len(df_existing)}"

    return True, "Data validation passed"


def run_export():
    """Load the new data the way the app does and write the derived exports."""
    try:
        logger.info("Running pipeline.export_derived to process updated data...")
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.export_derived"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=ROOT
        )
        
        if result.returncode == 0:
            logger.info("pipeline.export_derived completed successfully")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"pipeline.export_derived failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("pipeline.export_derived timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run pipeline.export_derived: {e}")
        return False


# Deployment layout (see docker-compose.yml): the app runs as the "dashboard"
# service behind Caddy, with data/ bind-mounted into the container.
# Restarting the service is what picks up the regenerated CSVs.
COMPOSE_SERVICE = "dashboard"


def _compose_command():
    """Return a working Compose CLI invocation, or None if Docker is unavailable."""
    for candidate in (["docker", "compose"], ["docker-compose"]):
        try:
            result = subprocess.run(
                candidate + ["version"], capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


def _compose_service_running(compose):
    """True if the service currently has a running container."""
    try:
        result = subprocess.run(
            compose + ["ps", "--quiet", COMPOSE_SERVICE],
            cwd=ROOT, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        return False
    return bool(result.stdout.strip())


def _restart_compose_service():
    """Restart the Dash container so it loads the new data. Returns None if not applicable."""
    if not os.path.exists(os.path.join(ROOT, "docker-compose.yml")):
        logger.info("No docker-compose.yml here - not a Compose deployment")
        return None

    compose = _compose_command()
    if compose is None:
        logger.warning("Docker CLI not available (or no permission) - cannot use Compose")
        return None

    logger.info(f"Restarting Compose service '{COMPOSE_SERVICE}'...")
    try:
        result = subprocess.run(
            compose + ["restart", COMPOSE_SERVICE],
            cwd=ROOT, capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        logger.error("Compose restart timed out after 180 seconds")
        return False

    if result.returncode != 0:
        logger.error(f"Compose restart failed (exit {result.returncode}): {result.stderr.strip()}")
        return False

    # `compose restart` exits 0 even when the service has no container at all,
    # so confirm something is actually running before reporting success.
    if not _compose_service_running(compose):
        logger.error(f"Service '{COMPOSE_SERVICE}' is not running after the restart")
        logger.info(f"Start it manually: docker compose up -d {COMPOSE_SERVICE}")
        return False

    logger.info(f"Service '{COMPOSE_SERVICE}' restarted with the new data")
    return True


def _reload_host_gunicorn():
    """Fallback for a bare host-side Gunicorn (no container): graceful worker reload."""
    patterns = [
        "gunicorn.*wsgi:application",
        f"gunicorn.*{ROOT}",
    ]

    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            continue

        master_pid = result.stdout.strip().split('\n')[0]  # Get first (master) process
        logger.info(f"Found Gunicorn master process: {master_pid}")
        try:
            subprocess.run(["kill", "-HUP", master_pid], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send HUP signal to {master_pid}: {e}")
            return False

        logger.info(f"Sent HUP signal to Gunicorn master process {master_pid}")
        # Note: with --preload the app would live in the master process, so HUP restarts
        # the workers but does NOT re-read the data - a full restart is needed there.
        logger.info("Gunicorn workers reloading (a --preload master needs a full restart)")
        return True

    logger.error("Could not locate a Gunicorn master process")
    return False


def restart_app():
    """Restart the Dash application after a data update (Compose service, else Gunicorn)."""
    try:
        compose_result = _restart_compose_service()
        if compose_result is not None:
            return compose_result

        logger.info("Falling back to a host-side Gunicorn reload...")
        if _reload_host_gunicorn():
            return True

        logger.info(f"Manual restart required: docker compose restart {COMPOSE_SERVICE}")
        return False

    except Exception as e:
        logger.error(f"Failed to restart application: {e}")
        logger.info(f"Manual restart: docker compose restart {COMPOSE_SERVICE}")
        return False


def update_data(force=False):
    """Main function to update laureates data from API."""
    logger.info("=" * 80)
    logger.info("Starting Nobel Prize data update")
    logger.info("=" * 80)

    try:
        # Load existing data for comparison (if available)
        df_existing = None
        try:
            df_existing = pd.read_csv(LAUREATES_CSV, sep=";", encoding="UTF-8")
            logger.info(f"Loaded existing data: {len(df_existing)} records")
        except FileNotFoundError:
            logger.info("No existing data file found for comparison")
        except Exception as e:
            logger.warning(f"Error loading existing data: {e}")

        # Fetch new data from API
        logger.info("Fetching data from Nobel Prize API...")
        df_laureates_new = get_all_laureates_data()

        if df_laureates_new is None:
            logger.error("Failed to retrieve laureates data from API")
            return False

        logger.info(f"Successfully retrieved {len(df_laureates_new)} laureates from API")

        # Validate the new data
        is_valid, validation_message = validate_laureates_data(df_laureates_new, df_existing)

        if not is_valid:
            logger.error(f"Data validation failed: {validation_message}")
            logger.error("Skipping data update to preserve data integrity")
            return False

        logger.info(f"Data validation passed: {validation_message}")

        # Save raw data to files. Keep the previous file: if the new data breaks the
        # loading code (export step below), it is put back, so a later container
        # restart (reboot, deploy) does not load data the app cannot handle.
        if os.path.exists(LAUREATES_CSV):
            paths.STATE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LAUREATES_CSV, LAUREATES_BACKUP)
        logger.info("Saving raw data to files...")
        success = save_laureates_data(df_laureates_new)

        if not success:
            logger.error("Failed to save laureates data")
            restore_previous_data()
            return False

        logger.info(f"Raw data saved successfully - {len(df_laureates_new)} records")

        # The rows are sorted by ID, so identical API data gives an identical file.
        # Skipping here keeps the announcement window (a run every five minutes)
        # from restarting the app -- with a short outage each time -- for nothing.
        raw_digest = file_digest(LAUREATES_CSV)
        if not force and raw_digest == read_processed_digest():
            save_update_timestamp()
            logger.info("API data unchanged since the last processed update - "
                        "skipping processing and restart")
            return True

        # Load the raw data the way the app does (and write the derived exports)
        processing_success = run_export()

        if not processing_success:
            logger.error("Data processing through pipeline.export_derived failed")
            restore_previous_data()
            return False

        # Update timestamp after complete success
        save_update_timestamp()
        
        logger.info("=" * 80)
        logger.info("Data pipeline update successful - restarting application...")
        logger.info("=" * 80)
        
        # Restart the application to load new data
        restart_success = restart_app()
        
        if restart_success:
            logger.info("Application restart completed successfully")
            write_processed_digest(raw_digest)
        else:
            logger.warning("Application restart may have failed - manual restart might be needed")
        
        return True

    except Exception as e:
        logger.error(f"Error during data update: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = update_data(force="--force" in sys.argv[1:])
    sys.exit(0 if success else 1)