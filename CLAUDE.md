# Nobel Data Dashboard (deploy2)

Plotly-Dash-Dashboard zu Nobelpreis-Daten (nbldata.org). Struktur und Modul-Split
sind im README dokumentiert; `plotdatagenerator.py` ist nur noch eine Fassade über
data.py / filters.py / figs_*.py / network.py.

## Memories

Repo-spezifische Memories liegen unter `.claude/memory/`, Index in
[.claude/MEMORY.md](.claude/MEMORY.md) — zu Sitzungsbeginn lesen.

## Arbeiten in diesem Repo

- Environment: `uv sync`, dann `uv run python app.py` (Details im README).
  `.venv/` nicht via Nextcloud syncen.
- `requirements.txt` ist aus `uv.lock` generiert — nie von Hand editieren
  (`uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt`).
- Datenpipeline: `python plotdatagenerator.py` regeneriert die abgeleiteten CSVs
  inkl. `df_edges.csv` (wird von update_data.py als Subprocess aufgerufen).
- Statischer Check vor Commits: `uv run ruff check .` (F821 hat beim Modul-Split
  echte Fehler gefunden).
