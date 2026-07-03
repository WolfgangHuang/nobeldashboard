---
name: dev-environment
description: Environment läuft über uv (pyproject.toml + uv.lock); .venv nicht via Nextcloud syncen
metadata:
  type: project
---

Das Projekt nutzt seit Juli 2026 **uv**: Abhängigkeiten in `pyproject.toml`
(nur direkte Deps, gepinnt), exakte Auflösung in `uv.lock`; `requirements.txt`
wird per `uv export` aus dem Lock generiert (nicht von Hand editieren).

**Why:** Vorher existierte auf diesem Rechner gar kein venv (VSCode zeigte auf
ein nicht-synchronisiertes `.venv/`), und die alte requirements.txt war ein
pip-freeze mit 70 transitiven Paketen.

**How to apply:** `uv sync` erzeugt `.venv/`, `uv run python app.py` startet die
App. Achtung Nextcloud: `.venv/` vom Sync ausschließen oder via
`UV_PROJECT_ENVIRONMENT` außerhalb des Ordners anlegen — sonst werden tausende
Dateien auf 3 Rechner synchronisiert. Hinweis: `mcp`/`starlette` im Lock sind
legitime transitive Deps von dash 4.3 (eingebaute MCP-Integration), kein Ballast.
