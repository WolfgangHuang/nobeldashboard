# Handoff: nbldata.org Dashboard Redesign

## Overview
A visual refresh of the nbldata.org Nobel-laureate analytics dashboard. The redesign keeps the existing **Dash + Plotly + Dash Mantine Components (DMC / Mantine)** stack and reworks the look & feel: a recurring **"Nobel-Spektrum"** identity (six fixed, clearly distinguishable category colors — one per prize), a bolder header + KPI treatment, a compact filter toolbar, a redesigned collapsible sidebar, a bento-style chart grid, and a unified Plotly theme. Light and dark modes are both first-class.

## About the Design Files
The file in this bundle (`NBLData Redesign.dc.html`) is a **design reference created in HTML** — a prototype showing the intended look, layout, color, type and chart styling. **It is not production code to copy directly.** `support.js` is only the rendering runtime for the prototype and is **not** part of the deliverable — ignore it for implementation.

Your task is to **recreate these designs in the existing nbldata.org codebase** (Python / Dash with `dash-mantine-components` and Plotly figures), using its established patterns: DMC components for layout/controls, Plotly figures for charts, a Mantine theme object for tokens, and a shared Plotly template for chart styling. Do not ship the HTML.

The prototype is a single pannable "canvas" containing, top to bottom: (01) direction & rationale, (02) the two chosen color palettes, (03) typography, (04) component closeups, (05) the full Overview page in light **and** dark, (06) the 2025 Prizes page, (07) the Plotly theme examples. Sections 01–04 are reference; **sections 05–07 are the screens to build.**

## Fidelity
**High-fidelity.** Colors, typography, spacing, radii and chart styling are final. Recreate pixel-faithfully using DMC/Mantine + Plotly equivalents. Where a raw pixel value doesn't map cleanly to a Mantine token, pick the nearest token rather than hard-coding.

---

## Design Tokens

### Category colors — the "Nobel-Spektrum" (one per prize)
Six fixed colors, used everywhere a prize category appears (chips, chart series, KPI accents, the logo). **Light mode = "Jewel"** (deeper, muted), **Dark mode = "Neon"** (brighter, on dark ground).

| Prize        | Light (Jewel) | Dark (Neon) |
|--------------|---------------|-------------|
| Medicine     | `#B23A48`     | `#FF6B6E`   |
| Physics      | `#2A4D9B`     | `#6E8BFF`   |
| Chemistry    | `#C2682A`     | `#FF8A3D`   |
| Economics    | `#6D3A9C`     | `#B97BE6`   |
| Literature   | `#B08A1E`     | `#F2C04B`   |
| Peace        | `#1F7A5A`     | `#4CD389`   |

Canonical order (logo, legends, distribution lists): Medicine, Physics, Chemistry, Economics, Literature, Peace.

### Neutrals
**Light mode**
- Page background: `#EEF1F4` (the blue-gray the user liked — keep it)
- Surface / cards: `#FFFFFF`
- Header background: `#FFFFFF` (was black — now light; separated from body by a 1px hairline `#E6E9ED`)
- Card border: `#E4E7EC`; subtle inner dividers `#EFF1F4` / `#F0F2F5`
- Text primary: `#14181E`; secondary `#5B636E`; muted/labels `#9AA1AB`; faint label `#A7AEB8`
- Track / slider rail: `#EAECF0`

**Dark mode**
- Page background: `#0C0F14`
- Surface / cards: `#111722`
- Card border: `#20262F`; faint label text `#5B626C`
- Text primary: `#FFFFFF`; secondary `#9BA3AE`; muted `#6B7280`
- Active nav background: `#241A33` (with `#B97BE6` accent)
- Distribution track: `#1B2027`

### Typography
- **Display / headings:** `Space Grotesk` (600/700). Used for page titles, card titles, KPI numbers.
- **UI / body:** `IBM Plex Sans` (400/500/600). Nav, labels, chips, descriptions.
- **Numbers / mono accents:** `IBM Plex Mono` (400/500/600). Eyebrow labels, axis ticks, ranges, hex codes, small stats. Apply `font-variant-numeric: tabular-nums` to KPI numbers.
- **Icons:** Material Symbols Rounded (see Assets). The codebase may instead use its existing icon set — match the glyph meaning.

Type scale seen in the design:
- Page title (hero): 21px / 700 Space Grotesk
- Section eyebrow: 13px / 600 IBM Plex Mono, letter-spacing .14em, uppercase, color `#A7AEB8`
- Card title: 14–15px / 600 Space Grotesk
- KPI number: 36px (in-page) / 42px (closeup), 600 Space Grotesk, tabular-nums
- KPI label: 11px / 600 IBM Plex Sans, uppercase, letter-spacing .03em
- Nav item: 13–14px / 500 (600 when active) IBM Plex Sans
- Nav section label ("ANALYSE", "WERKZEUGE"): 10px / 600 IBM Plex Mono, letter-spacing .12em
- Chip: 12–13px / 600 IBM Plex Sans
- Body/description: 13–14px / 400 IBM Plex Sans, line-height ~1.5

### Spacing, radius, shadow
- Card padding: 16–20px. Gaps between cards: 14–16px.
- Border radius: cards 14px; large page frame 18px; pills/chips 999px; nav items 10px; KPI/swatch 8px; inputs 999px (search) / 10px (icon button).
- Card shadow (light): `0 1px 2px rgba(16,24,40,.04)` for inner cards; `0 1px 3px rgba(16,24,40,.06)` for standalone. Dark mode: no shadow, rely on `#20262F` borders.
- Active nav indicator: a 3px-wide rounded bar pinned to the left edge of the item, in the active category color, with a tinted background (`#F4F0FB` light / `#241A33` dark).

---

## Screens / Views

### 1. Global frame (applies to all pages)
**Layout:** Fixed left **sidebar** (212px, collapsible to ~72px icon-rail) + top **header** bar + scrollable **main** content area on the `#EEF1F4` background.

**Header (light):** white background, 1px `#E6E9ED` hairline below it. Left: 6-stripe spectrum logo mark (six 8×26px rounded bars in canonical order, 3px gap) + wordmark "Nobel Laureate Data Dashboard" (21px Space Grotesk, `#14181E`) with a mono subtitle "1901–2025 · 1026 prizes · 1018 laureates" (`#9AA1AB`). Right: a pill search field ("Suche Laureate…", `#F2F4F7` bg), a 38×38 rounded icon button toggling light/dark (`dark_mode` / `light_mode` icon), and a "V2.0" mono badge (`#1F7A5A` on tinted green). **Dark header:** background `#111722`, white text, dark search/toggle, green badge `#4CD389`.

> Note: the original full-width rainbow band under the header was removed (felt too busy) — keep only the 1px hairline. The spectrum identity lives in the logo.

**Sidebar:** white (`#111722` dark), 1px right border. Logo mark + "nbldata" wordmark (16px/700 Space Grotesk) at top. Two grouped sections with mono labels:
- **ANALYSE:** Overview (active), 2025 Prizes, Geography, Demography, Time Analysis, Migration, Nominations
- **WERKZEUGE:** List Generator, Data & Refs

Each item = icon (19–20px) + label, 9–12px padding, 10px radius. Active item: tinted bg + 3px left accent bar in the active category color (Overview uses Economics purple — `#6D3A9C` light / `#B97BE6` dark — and a filled icon). Collapsed state: 44×44 icon tiles only, active tile keeps the tinted bg.

### 2. Overview page (screen 05) — primary page
Main column (padding ~22–24px) stacks:

**a. Filter toolbar** (compact, single row, wraps): white card, 14px radius. Contains, separated by 1px `#EAECF0` dividers:
- Six category **toggle chips** (pill, 999px). Selected = filled in that category color with a white `check` icon + white label. Deselected = white bg, `#E4E7EC` border, `#9AA1AB` label (the design shows Literature deselected in the closeup, all selected in the full page).
- A **gender segmented control** (female / male) — Mantine `SegmentedControl`, pill style, `#F2F4F7` track.
- A **year range slider** (1901–2025) — Mantine `RangeSlider`. The rail shows a left-to-right category gradient; thumbs are white with a 2px colored ring. Range label in mono above.

**b. KPI row:** four equal cards, 14px radius, each with a 4px left accent bar in a category color and a matching icon:
- **Prizes** — `#2A4D9B` (Physics) / `military_tech` — value **1026**, sub "excl. declined"
- **Laureates** — `#1F7A5A` (Peace) / `groups` — value **1018**, sub "incl. organisations"
- **Youngest** — `#B23A48` (Medicine) / `trending_down` — value **17**, name "Malala Yousafzai"
- **Oldest** — `#6D3A9C` (Economics) / `trending_up` — value **97**, name "John B. Goodenough"

Values are 36px Space Grotesk, tabular-nums.

**c. Bento chart grid** (12-column, 14px gaps):
- **span 7** — "Discipline · Gender · Country" **Sunburst** (clicking a segment filters the dashboard). Card title + sub + a "2025" tinted pill.
- **span 5** — a dark **"Spektrum-Verteilung"** card: per-category horizontal **bars** (label + mono count + a thin track with a category-colored fill). Counts shown: Medicine 227, Physics 226, Chemistry 195, Peace 142, Literature 120, Economics 96. Below it, a small **Gender donut**.
- **span 6** — **Ethnicity donut**.
- **span 6** — **Religion donut**.

### 3. 2025 Prizes page (screen 06)
Header row: title "Preisträger 2025" (24px Space Grotesk) + sub, and a dark "15 Laureaten" pill on the right. Below: a **3-column card grid** (16px gap), one card per category. Each card: 14px radius, white, with a **5px top accent bar** in the category color; inside: an icon + uppercase category label in the category color, the laureate name(s) in 18px Space Grotesk, and the motivation text (13px IBM Plex Sans, `#5B636E`) above a 1px top divider. Categories/colors/icons:
- Medicine `#B23A48` `vaccines` — Mary E. Brunkow · Fred Ramsdell · Shimon Sakaguchi — "for their discoveries concerning peripheral immune tolerance"
- Physics `#2A4D9B` `bolt` — John Clarke · Michel H. Devoret · John M. Martinis — "for the discovery of macroscopic quantum mechanical tunnelling in an electric circuit"
- Chemistry `#C2682A` `science` — Susumu Kitagawa · Richard Robson · Omar M. Yaghi — "for the development of metal–organic frameworks"
- Literature `#B08A1E` `menu_book` — László Krasznahorkai — "for his compelling and visionary oeuvre that reaffirms the power of art"
- Peace `#1F7A5A` `handshake` — María Corina Machado — "for her tireless work promoting democratic rights for the people of Venezuela"
- Economic Sciences `#6D3A9C` `trending_up` — Joel Mokyr · Philippe Aghion · Peter Howitt — "for the theory of sustained growth through creative destruction"

> 2025 laureate data above is placeholder/illustrative — wire it to the real data source.

---

## Plotly theme (screen 07) — unified template
Build **one Plotly template** (a `go.layout.Template`, or a shared `_apply_theme(fig, dark)` helper) and apply to every figure. Settings observed in the prototype:

- `paper_bgcolor` / `plot_bgcolor`: transparent (`rgba(0,0,0,0)`) — cards provide the background.
- Global font: `IBM Plex Sans`; chart centers/headlines use `Space Grotesk`; axis ticks use `IBM Plex Mono` 10px.
- **No chart chrome:** `displayModeBar: false`, `responsive: true`. Hide x-grid; y-grid only, faint (`#EDEFF2` light). `zeroline: false` everywhere. Tick color `#9AA1AB`.
- Margins tight (≈ `t/b/l/r` 6–36px depending on chart).
- **Donuts:** `hole: 0.64`, `sort: false`, `direction: 'clockwise'`, slice separators = 3px line in the surface color (`#FFFFFF` light / `#1B2027` dark), labels outside (`label+percent`), a center annotation label. Outside-text color `#6B7280` light / `#9BA3AE` dark.
- **Sunburst:** `branchvalues: 'total'`, segment border 2px in surface color, white radial inside-text; inner ring = the six categories in category colors, outer ring = countries inheriting the parent category color.
- **Stacked bar (Prizes by country per year):** `barmode: 'stack'`, `bargap: 0.15`, horizontal legend above the plot, series colored by the country's mapping (USA→Physics blue, UK→Medicine red, Germany→Chemistry orange, France→Economics purple, Japan→Peace green in the demo — replace with the project's real country→color logic).
- Use the **Jewel** colorway in light mode and the **Neon** colorway in dark mode (same category→color tables above).

Suggested implementation: define `SPECTRUM_LIGHT` / `SPECTRUM_DARK` dicts in Python, register a Plotly template per mode, and set `fig.update_layout(template=...)` + `colorway` centrally so individual figure code stays clean.

---

## Interactions & Behavior
- **Theme toggle** in the header switches light/dark: swaps neutrals, swaps the category colorway (Jewel↔Neon), and re-renders Plotly figures with the matching template. Persist the choice (e.g. `dcc.Store` + `localStorage`).
- **Sidebar collapse:** toggles between 212px labeled and ~72px icon-rail; active state preserved.
- **Category chips:** multi-select toggle; drives a filter applied to all figures + KPIs.
- **Gender segmented control** and **year RangeSlider:** also feed the global filter.
- **Sunburst click:** filters the dashboard to the clicked discipline/country (Plotly `clickData` → callback).
- Hover: Plotly tooltips use `IBM Plex Sans`. Nav/chip hover states should lightly tint background.

## State Management (Dash)
- `theme` (light/dark) — `dcc.Store`, mirrored to localStorage.
- `selected_categories` (list of the six), `gender` (female/male/all), `year_range` ([1901, 2025]) — `dcc.Store` or component values feeding a single filtering callback.
- `sidebar_collapsed` (bool).
- Figures are outputs of callbacks that read the filter state + theme and return themed `go.Figure` objects. KPI values recompute from the filtered dataframe.

## Assets
- **Icons:** Material Symbols Rounded (Google Fonts) in the prototype: `dashboard, emoji_events, public, groups, calendar_month, swap_horiz, campaign, table_chart, storage, military_tech, trending_up, trending_down, search, dark_mode, light_mode, check, menu, vaccines, bolt, science, menu_book, handshake`. Use the codebase's existing icon library if it has one; otherwise Material Symbols via a font `<link>`.
- **Fonts:** Space Grotesk, IBM Plex Sans, IBM Plex Mono (Google Fonts). Load via Dash `external_stylesheets` or `assets/`.
- **Logo:** the 6-stripe spectrum mark is pure CSS/HTML (six rounded bars) — recreate as a small component, no image asset needed. Keep it **six** stripes consistently everywhere.
- No raster images or photos are used.

## Files
- `NBLData Redesign.dc.html` — the full hi-fi design reference (open in a browser to inspect every screen, color, and the live Plotly charts). Charts are defined in the inline `class Component` script near the bottom — that JS is the **source of truth for the Plotly figure config** (donut/sunburst/bar settings, colors, fonts) and is the most useful part to port to Python Plotly.
- `support.js` — prototype runtime only; **ignore for implementation.**
