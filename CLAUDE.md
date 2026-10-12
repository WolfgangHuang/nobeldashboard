# Nobel Data Dashboard (deploy2)

Plotly-Dash-Dashboard zu Nobelpreis-Daten (nbldata.org). Struktur im README:
`app.py` + Paket `nbldash/` (Fassade `nbldash/plotdatagenerator.py` über data.py /
filters.py / figs/ / network.py), Host-Skripte in `pipeline/`, Daten unter
`data/{raw,curated,derived,state}` — Pfade nur über `nbldash/paths.py`.

## Memories

Repo-spezifische Memories liegen unter `.claude/memory/`, Index in
[.claude/MEMORY.md](.claude/MEMORY.md) — zu Sitzungsbeginn lesen.

## Arbeiten in diesem Repo

- Environment: `uv sync`, dann `uv run python app.py` (Details im README).
  `.venv/` nicht via Nextcloud syncen.
- `requirements.txt` ist aus `uv.lock` generiert — nie von Hand editieren
  (`uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt`).
- Datenpipeline: Skripte als Module aus dem Root starten (`python -m
  pipeline.update_data`, `python -m pipeline.export_derived`). Die Kanten werden
  beim Import aus `nominations_full.csv` gebaut; es gibt kein `df_edges.csv` mehr.
- Was der Cron schreibt (raw/df_laureates.csv, derived/, state/), ist gitignored.
  Ort per `NBL_DATA_DIR` umstellbar (unter Dokploy außerhalb des Checkouts, siehe README).
- Kein `--preload` für Gunicorn (Polars-Deadlock nach fork, siehe Dockerfile).
- Statischer Check vor Commits: `uv run ruff check .` (F821 hat beim Modul-Split
  echte Fehler gefunden).
