---
name: dev-environment
description: Kein .venv im Repo trotz VSCode-Verweis; wie App/Tests hier laufen
metadata:
  type: project
---

Auf diesem Linux-Rechner existiert **kein** `.venv/` im Repo, obwohl
`.vscode/settings.json` auf `.venv/bin/python` zeigt (vermutlich wird das venv
nicht über Nextcloud synchronisiert). System-Python (3.14) hat keine
Dashboard-Abhängigkeiten.

**Why:** Ohne dieses Wissen schlägt jeder Test-/Verifikationslauf mit
`ModuleNotFoundError: plotly` fehl.

**How to apply:** Für Verifikation ein Wegwerf-venv außerhalb des Repos anlegen
(`python -m venv <scratch>/venv && pip install -r requirements.txt` — die
Requirements installieren sauber, Stand Juli 2026) und damit `python app.py`
bzw. `gunicorn --preload -w 2 app:server` starten. Kein venv IM Repo anlegen —
Nextcloud würde tausende kleine Dateien auf 3 Rechner synchronisieren.

Hinweis: `plotdatagenerator.py` ist seit Juli 2026 nur noch eine Fassade über
data.py/filters.py/figs_*.py/network.py; als Skript ausgeführt regeneriert es
die abgeleiteten CSVs inkl. `df_edges.csv` (Details in Git-History).
