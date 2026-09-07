"""
Nobel Prize Data Update Script
Standalone script to update data from Nobel Prize API
Designed to be run via cron job or systemd timer
"""

import sys
import os
import logging
import subprocess
import pandas as pd
from datetime import datetime

# Add current directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
os.chdir(current_dir)

from utils import get_all_laureates_data, save_laureates_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_update')


def save_update_timestamp():
    """Save the current timestamp to file."""
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    try:
        with open("last_api_update.txt", 'w') as f:
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


def run_plotdatagenerator():
    """Execute plotdatagenerator.py to process raw data into enriched datasets."""
    try:
        logger.info("Running plotdatagenerator.py to process updated data...")
        result = subprocess.run(
            [sys.executable, "plotdatagenerator.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=current_dir
        )
        
        if result.returncode == 0:
            logger.info("plotdatagenerator.py completed successfully")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"plotdatagenerator.py failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("plotdatagenerator.py timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run plotdatagenerator.py: {e}")
        return False


# Deployment layout (see docker-compose.yml): the app runs as the "dashboard"
# service behind Caddy, with this directory bind-mounted into the container.
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
            cwd=current_dir, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        return False
    return bool(result.stdout.strip())


def _restart_compose_service():
    """Restart the Dash container so it loads the new data. Returns None if not applicable."""
    if not os.path.exists(os.path.join(current_dir, "docker-compose.yml")):
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
            cwd=current_dir, capture_output=True, text=True, timeout=180
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
        f"gunicorn.*{current_dir}",
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
        # Note: with --preload the app lives in the master process, so HUP restarts
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


def update_data():
    """Main function to update laureates data from API."""
    logger.info("=" * 80)
    logger.info("Starting Nobel Prize data update")
    logger.info("=" * 80)

    try:
        # Load existing data for comparison (if available)
        df_existing = None
        try:
            df_existing = pd.read_csv("df_laureates.csv", sep=";", encoding="UTF-8")
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

        # Save raw data to files
        logger.info("Saving raw data to files...")
        success = save_laureates_data(df_laureates_new, output_dir=".")

        if not success:
            logger.error("Failed to save laureates data")
            return False

        logger.info(f"Raw data saved successfully - {len(df_laureates_new)} records")

        # Process the raw data through plotdatagenerator
        processing_success = run_plotdatagenerator()

        if not processing_success:
            logger.error("Data processing through plotdatagenerator failed")
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
        else:
            logger.warning("Application restart may have failed - manual restart might be needed")
        
        return True

    except Exception as e:
        logger.error(f"Error during data update: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = update_data()
    sys.exit(0 if success else 1)