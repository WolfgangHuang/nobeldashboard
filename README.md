# Nobel Prize Data Dashboard

A comprehensive interactive dashboard for exploring Nobel Prize data, built with Dash and Plotly. The dashboard provides visualizations covering laureate demographics, geographic distribution, historical trends, and nomination networks.

## Features

- Interactive visualizations with filtering by category, gender, and time range
- Geographic analysis: choropleth maps, city-level distributions, migration patterns
- Demographic breakdowns: gender, religion, ethnicity, age at award
- Time series analysis: prizes over time, publication-to-prize time gaps
- Nomination network graphs (based on historical nomination data with 50-year secrecy period)
- List generator with export functionality

## Project Structure

```
├── app.py                     # Dash app: layout, callbacks, routing (assets/ must sit next to it)
├── wsgi.py                    # Gunicorn entry point (wsgi:application)
├── assets/                    # Served publicly by Dash: CSS, JS, favicons, images
├── nbldash/                   # Dashboard package (imported by app.py)
│   ├── paths.py               # Where all data files live (standard library only)
│   ├── data.py                # Loading, cleaning, enrichment, nomination edges (runs at import)
│   ├── filters.py             # standard_filter / extended_filter + chip-state helpers
│   ├── network.py             # Nominations network (filtering, graph, cytoscape, map)
│   ├── overview_stats.py      # Overview KPIs
│   ├── figs/                  # Figure generators: geography, demography, time_age,
│   │                          #   migration, prizemoney
│   ├── plotdatagenerator.py   # Facade re-exporting all of the above (`pdg` in app.py)
│   ├── plot_config.py         # PlotConfig class + per-plot configuration
│   ├── config.py              # App settings + color constants
│   ├── layouts.py             # Widget layout builders (plot shells, network widget)
│   ├── filters_ui.py          # Filter drawer renderers
│   └── theme.py               # Nobel-Spektrum design system (colors, templates)
├── pipeline/                  # Runs on the host, not in the container
│   ├── scheduled_update.py    # Cron entry point: owns the schedule
│   ├── update_data.py         # API fetch → validate → export → restart
│   ├── nobel_api.py           # Nobel Prize API client
│   ├── export_derived.py      # Loads the data like the app, writes data/derived/
│   ├── scrape_nominations.py  # Nomination archive scraper (run by hand)
│   └── logs.py                # Rotating log in data/state/
├── data/
│   ├── raw/                   # df_laureates.csv (API, written by cron, not in git),
│   │                          #   nominations_full.csv (scraper, in git)
│   ├── curated/               # Maintained by hand: corrections, ethnicity, religion, ...
│   ├── derived/               # Exports for analysis (not in git, not read by the app)
│   └── state/                 # Timestamps, lock, log (not in git)
├── design/                    # Design sources and unused images (not served)
└── Dockerfile, docker-compose.yml, Caddyfile
```

Everything written by the update cron lives in git-ignored files, so `git pull`
on the server never conflicts with fresh data.

The nomination edges are built from `nominations_full.csv` at import (~0.2 s);
there is no `df_edges.csv` any more that could fall behind the nominations.

## Key Dependencies

- dash / dash-mantine-components - Web framework and UI components
- plotly - Interactive visualizations
- pandas - Data manipulation (laureate data)
- polars - High-performance data processing (nominations, list generator)
- networkx / pygraphviz - Network graph generation
- yfinance - Prize money currency conversion

## Running the Application

Environment (uv — dependencies are declared in `pyproject.toml`, exact versions in `uv.lock`):
```bash
uv sync                               # creates .venv/ and installs the locked dependencies
uv run python -m pipeline.nobel_api   # first run only: fetch data/raw/df_laureates.csv
uv run python app.py                  # run inside the environment
```
`requirements.txt` is generated from the lock (`uv export --format requirements-txt
--no-dev --no-hashes -o requirements.txt`) for pip-based deploys — don't edit it by hand.

Note: if this folder is synced (Nextcloud), exclude `.venv/` from sync, or place the
environment elsewhere via `UV_PROJECT_ENVIRONMENT=~/.venvs/nobel-dashboard`.

Development:
```bash
python app.py                          # binds 127.0.0.1:8050, debug off
DEBUG=true HOST=0.0.0.0 python app.py  # opt-in: Werkzeug debugger + external binding
```

Note: `DEBUG=true` enables the Werkzeug interactive debugger (arbitrary code
execution) — never combine it with an externally reachable host.

Production (Gunicorn):
```bash
gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:8050 app:server
```

Do not add `--preload`: polars deadlocks in a forked worker once its thread pool
has run in the master (building the edge table at import does that), so every
nominations/network callback would hang. Each worker loads the app itself
(~450 MB RSS); the `gthread` workers keep a few slow requests from blocking the app.

## Deployment (Docker Compose)

`docker-compose.yml` runs the app behind Caddy (HTTPS, security headers). The image
contains the code only; `data/` is bind-mounted read-only, so:

```bash
docker compose up -d --build        # after a code change (git pull)
docker compose restart dashboard    # load new data (update_data.py does this)
```

The `dashboard` service has a healthcheck on `/healthz`; Caddy starts once it is
healthy. Container logs are capped at 3 × 10 MB per service.

A fresh server needs the data before the first start: `uv sync`, then
`uv run python -m pipeline.update_data --force` (fetches the API data and writes
`data/state/`; the restart step fails harmlessly while nothing runs yet), then
`docker compose up -d --build`.

## Data Updates

To update laureate data from the API (from the project root):
```bash
python -m pipeline.update_data
```

This script:
1. Fetches current data from the Nobel Prize API
2. Validates the response
3. Saves it to `data/raw/df_laureates.csv`
4. Runs `pipeline.export_derived`, which loads the data exactly like the app — if
   that fails, the app is not restarted and keeps serving the old data
5. Restarts the Compose service `dashboard` (falls back to a Gunicorn HUP when
   the deployment is not containerised)

Steps 4 and 5 only run when `df_laureates.csv` differs from the data the last
complete run processed (SHA-256 in `data/state/last_processed_laureates.sha256`);
otherwise only the timestamp is refreshed. This keeps the announcement windows,
with a run every five minutes, from restarting the app each time. `--force`
processes and restarts regardless, e.g. after changing the loading code.

Run it on the host, not inside the container: it needs the Docker CLI for step 5.
The log goes to `data/state/data_update.log` (rotated at 1 MB, three old copies).

New nominations: run `pipeline/scrape_nominations.py` by hand (`python -m
pipeline.scrape_nominations`); it writes to `data/derived/`. After checking the
result, copy it to `data/raw/nominations_full.csv` and commit it.

### Scheduling

`pipeline/scheduled_update.py` is the cron entry point and owns the schedule, so
the crontab holds a single line:

```
*/5 * * * * cd /path/to/repo && .venv/bin/python -m pipeline.scheduled_update >> ~/cron-nbldata.log 2>&1
```

It exits within milliseconds unless an update is due:

- once a day at `DAILY_TIME` (08:00)
- every five minutes inside a Nobel announcement window — from the announced
  time until `WINDOW_HOURS` (3h) later, on the announcement days

All times are `Europe/Stockholm`. The server clock runs on UTC, so putting the
announcement times into the crontab directly would fire them two hours late
during CEST. Converting inside the script also survives the winter-time switch.

The announcement days are derived per year from `ANNOUNCEMENT_PATTERN`
(Medicine on the first Monday in October, Physics through Peace Tuesday to Friday,
Economics the following Monday), so nothing needs a yearly refresh. Should the
Nobel Foundation deviate (check https://www.nobelprize.org/press/), add the dates
to `ANNOUNCEMENT_OVERRIDES`.

```bash
python -m pipeline.scheduled_update --check   # print the decision, change nothing
python -m pipeline.scheduled_update --force   # update regardless of schedule (passes --force on)
```

A file lock prevents a slow run from overlapping with the next cron slot.

## License

Data sourced from the official Nobel Prize API and nomination archive.
