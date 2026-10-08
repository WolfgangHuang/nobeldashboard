---
name: deployment-server
description: Deploy-Topologie nbldata.org — Hetzner-VPS, Docker Compose + Caddy, Git-Remote als Deploy-Pfad; Serveruhr läuft auf UTC
metadata:
  type: project
---

Live läuft nbldata.org auf einem Hetzner-VPS (4 GB): SSH-Alias `Hetzner-WH`
(`dashboardadmin@37.27.252.120`, Login per Passwort, kein Key hinterlegt —
Zugriff geht praktisch über VSCode Remote-SSH). App-Verzeichnis `~/nbldata_app`,
dort `docker compose up -d --build` mit den Services `dashboard` (Gunicorn) und
`caddy`. Das Verzeichnis ist per Bind-Mount `./:/app` im Container, deshalb
wirken Datei-Änderungen ohne Rebuild — Dependency-Änderungen aber **nur** mit
`--build`.

Deploy-Weg seit 7. September 2026: GitHub-Repo
https://github.com/WolfgangHuang/nobeldashboard (public) → auf dem Server
`git pull`. Vorher lag die Deploy-Schicht (Dockerfile/Compose/Caddyfile) nur auf
dem Server und war nie im Repo. Der alte Repo-Stand vor dem Force-Push hängt am
Tag `pre-refactor-2025-07`.

**Die Serveruhr läuft auf UTC**, die Verkündungszeiten sind Stockholmer Zeit —
deshalb liegt die Zeitlogik in `scheduled_update.py` und nicht im Crontab.

**Why:** Weder Serveradresse noch die Force-Push-Historie noch die UTC-Falle
stehen im Repo; ohne das wird die Zeitzone beim nächsten Crontab-Eintrag wieder
falsch und der Backup-Tag nicht gefunden.

**How to apply:** Nach Dependency-Änderungen immer `docker compose up -d --build`
— ein blosses `up` nimmt stumm das alte Image (genau das kostete beim ersten
Deploy eine Runde: `ModuleNotFoundError: dash_cytoscape` trotz korrekter
requirements.txt). `update_data.py` und `scheduled_update.py` laufen auf dem
**Host** im dortigen `.venv`, nicht im Container: Schritt 5 braucht das
Docker-CLI. Da die generierten CSVs im Repo getrackt sind, vor einem Pull
`git checkout -- '*.csv'`. Siehe auch [[dev-environment]] und
[[review-2026-07-offene-punkte]].
