---
name: review-2026-07-offene-punkte
description: Nach dem großen Review vom 3. Juli 2026 noch offene Punkte + bewusste Entscheidungen
metadata:
  type: project
---

Beim Review/Refactoring am 3. Juli 2026 (Phasen 0–4, siehe Git-History ab
Initial-Commit e33a632) blieben offen:

1. **Visueller Dark-Mode-Check im Browser steht aus** — die Theme-Swaps sind nur
   programmatisch verifiziert (Roundtrip-Tests). Besonders prüfen: Globen, Karten
   (carto-darkmatter), Data-Seite, Netzwerk-Verhalten nach cola-Umstellung.
2. **Live-Server (nbldata.org) läuft noch auf altem Stand** — Deploy des neuen
   Codes + Umstellung auf `gunicorn --preload` steht aus.
3. **Optionale Folgearbeiten**, bewusst nicht umgesetzt: app.py-Callbacks in
   Feature-Module (register_callbacks-Muster; Begründung im Phase-4-Commit),
   Debounce für den Live-Nominierungszähler, Caching-Layer (nach dem
   df_edges-Fix kaum noch nötig).

**Nutzer-Entscheidungen aus dem Review** (nicht aus Code ableitbar):
- Netzwerk: cola-Layout **endlich** (kein Live-Mitdriften mehr beim Ziehen) —
  bewusster Trade-off gegen die Dauer-CPU-Last.
- Altlasten wurden nach `_archive/` verschoben statt gelöscht.

**Why:** Ohne diese Notiz wirkt der Stand „fertig", obwohl Browser-Check und
Server-Deploy fehlen; die Folgearbeiten würden ggf. doppelt analysiert.

**How to apply:** Bei nächster Session zu diesem Repo zuerst fragen, ob
Browser-Check/Deploy erledigt sind, dann diese Memory aktualisieren/löschen.
Siehe auch [[dev-environment]].
