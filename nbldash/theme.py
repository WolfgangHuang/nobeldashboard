"""
Nobel-Spektrum design system
=============================

Single source of truth for the redesigned nbldata.org dashboard:
- the six fixed prize/category colors (Jewel for light mode, Neon for dark mode)
- neutral palettes for light and dark mode
- the Dash Mantine Components theme object (fonts, radii, color ramps)
- helpers to build a unified Plotly template per color scheme

See redesign/README.md for the full design specification.
"""

from __future__ import annotations

import plotly.graph_objects as go


# ------------------------------------------------------------------------------------------------
# Category colors — the "Nobel-Spektrum" (one per prize)
# ------------------------------------------------------------------------------------------------

# Canonical order used everywhere a category appears (logo, legends, distributions).
CATEGORY_ORDER = ["Medicine", "Physics", "Chemistry", "Economics", "Literature", "Peace"]

# Light mode = "Jewel" (deeper, muted)
SPECTRUM_LIGHT = {
    "Medicine":   "#B23A48",
    "Physics":    "#2A4D9B",
    "Chemistry":  "#C2682A",
    "Economics":  "#6D3A9C",
    "Literature": "#B08A1E",
    "Peace":      "#1F7A5A",
}

# Dark mode = "Neon" (brighter, on dark ground)
SPECTRUM_DARK = {
    "Medicine":   "#FF6B6E",
    "Physics":    "#6E8BFF",
    "Chemistry":  "#FF8A3D",
    "Economics":  "#B97BE6",
    "Literature": "#F2C04B",
    "Peace":      "#4CD389",
}

# Icons per category (Material Symbols Rounded glyph names; used in KPI cards & 2025 page).
CATEGORY_ICONS = {
    "Medicine":   "vaccines",
    "Physics":    "bolt",
    "Chemistry":  "science",
    "Economics":  "trending_up",
    "Literature": "menu_book",
    "Peace":      "handshake",
}


def spectrum(dark: bool = False) -> dict:
    """Return the active category->color mapping for the given color scheme."""
    return SPECTRUM_DARK if dark else SPECTRUM_LIGHT


def colorway(dark: bool = False) -> list:
    """Category colors in canonical order — used as the Plotly colorway."""
    table = spectrum(dark)
    return [table[c] for c in CATEGORY_ORDER]


# ------------------------------------------------------------------------------------------------
# Neutrals
# ------------------------------------------------------------------------------------------------

NEUTRALS_LIGHT = {
    "page_bg":        "#EEF1F4",
    "surface":        "#FFFFFF",
    "header_bg":      "#FFFFFF",
    "hairline":       "#E6E9ED",
    "card_border":    "#E4E7EC",
    "divider":        "#EFF1F4",
    "text_primary":   "#14181E",
    "text_secondary": "#5B636E",
    "text_muted":     "#9AA1AB",
    "text_faint":     "#A7AEB8",
    "track":          "#EAECF0",
    "search_bg":      "#F2F4F7",
    "nav_active_bg":  "#F4F0FB",
    "grid":           "#EDEFF2",
}

NEUTRALS_DARK = {
    "page_bg":        "#0C0F14",
    "surface":        "#111722",
    "header_bg":      "#111722",
    "hairline":       "#20262F",
    "card_border":    "#20262F",
    "divider":        "#1B2027",
    "text_primary":   "#FFFFFF",
    "text_secondary": "#9BA3AE",
    "text_muted":     "#6B7280",
    "text_faint":     "#5B626C",
    "track":          "#1B2027",
    "search_bg":      "#1B2027",
    "nav_active_bg":  "#241A33",
    "grid":           "#1B2027",
}


def neutrals(dark: bool = False) -> dict:
    return NEUTRALS_DARK if dark else NEUTRALS_LIGHT


# ------------------------------------------------------------------------------------------------
# Fonts
# ------------------------------------------------------------------------------------------------

FONT_DISPLAY = "'Space Grotesk', sans-serif"   # headings, KPI numbers, card titles
FONT_BODY = "'IBM Plex Sans', sans-serif"      # nav, labels, chips, descriptions
FONT_MONO = "'IBM Plex Mono', monospace"       # eyebrows, axis ticks, ranges, stats

# Google Fonts + Material Symbols — added to the Dash external_stylesheets.
GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@400;500;600;700&"
    "family=IBM+Plex+Sans:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600&"
    "family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap"
)


# ------------------------------------------------------------------------------------------------
# Mantine theme
# ------------------------------------------------------------------------------------------------

def _ramp(base: str) -> list:
    """
    Build a 10-shade Mantine color ramp around a base hex color.

    Mantine expects exactly 10 shades (index 0 = lightest, 9 = darkest); the
    default "filled" shade is index 6. We anchor the base at index 6 and derive
    the rest by blending towards white (lighter) and black (darker).
    """
    base = base.lstrip("#")
    r, g, b = int(base[0:2], 16), int(base[2:4], 16), int(base[4:6], 16)

    def mix(c1, c2, t):
        return round(c1 + (c2 - c1) * t)

    def hexify(rgb):
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    # tint factors towards white for indices 0..5, base at 6, shades towards black 7..9
    tints = [0.85, 0.70, 0.55, 0.40, 0.26, 0.13]   # -> indices 0..5
    shades = [0.12, 0.26, 0.42]                     # -> indices 7..9

    ramp = []
    for t in tints:
        ramp.append(hexify((mix(r, 255, t), mix(g, 255, t), mix(b, 255, t))))
    ramp.append(hexify((r, g, b)))                  # index 6 = base
    for t in shades:
        ramp.append(hexify((mix(r, 0, t), mix(g, 0, t), mix(b, 0, t))))
    return ramp


# Mantine custom-color names per category (used in components via color="medicine" etc.)
CATEGORY_MANTINE_COLORS = {cat: cat.lower() for cat in CATEGORY_ORDER}


def build_mantine_theme() -> dict:
    """
    The DMC MantineProvider theme object. Light/dark neutrals are handled by
    Mantine's color-scheme + the redesign CSS; here we set fonts, radii and the
    six category color ramps (Jewel anchors — Neon is applied to figures, while
    DMC dark mode lightens components automatically).
    """
    colors = {name: _ramp(SPECTRUM_LIGHT[cat]) for cat, name in CATEGORY_MANTINE_COLORS.items()}

    return {
        "fontFamily": FONT_BODY,
        "fontFamilyMonospace": FONT_MONO,
        "headings": {
            "fontFamily": FONT_DISPLAY,
            "fontWeight": "600",
        },
        "defaultRadius": "md",
        "primaryColor": "economics",   # purple — the Overview accent
        "primaryShade": {"light": 6, "dark": 5},
        "colors": colors,
        "radius": {
            "md": "14px",
            "lg": "18px",
        },
    }


# ------------------------------------------------------------------------------------------------
# Plotly template (used in step 2 — graphics restyling)
# ------------------------------------------------------------------------------------------------

def build_plotly_template(dark: bool = False) -> go.layout.Template:
    """
    One unified Plotly template per color scheme. Transparent backgrounds (cards
    provide the surface), no chart chrome, faint y-grid only, mono axis ticks.
    """
    n = neutrals(dark)
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=colorway(dark),
        font=dict(family=FONT_BODY, color=n["text_secondary"], size=12),
        title=dict(font=dict(family=FONT_DISPLAY, size=15, color=n["text_primary"])),
        margin=dict(t=24, b=24, l=36, r=12),
        hoverlabel=dict(
            font=dict(family=FONT_BODY, size=12),
            bgcolor=n["surface"],
            bordercolor=n["card_border"],
        ),
        legend=dict(font=dict(family=FONT_BODY, size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(family=FONT_MONO, size=10, color=n["text_muted"]),
            linecolor=n["card_border"],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=n["grid"],
            zeroline=False,
            tickfont=dict(family=FONT_MONO, size=10, color=n["text_muted"]),
            linecolor=n["card_border"],
        ),
    )
    return template


# Cache the two template objects. IMPORTANT: the template must be embedded as an
# object in the figure (not referenced by registered name) — plotly.js in the browser
# does not know our server-side pio.templates names, so a bare name string is ignored.
_TEMPLATE_CACHE = {}


def get_template(dark: bool = False) -> go.layout.Template:
    if dark not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[dark] = build_plotly_template(dark)
    return _TEMPLATE_CACHE[dark]


def apply_theme(fig: go.Figure, dark: bool = False) -> go.Figure:
    """Apply the unified template + colorway to an existing figure in place."""
    fig.update_layout(template=get_template(dark), colorway=colorway(dark))
    return fig


# ------------------------------------------------------------------------------------------------
# Figure neutrals (light <-> dark pairs)
# ------------------------------------------------------------------------------------------------
# Some figure elements cannot come from the transparent template: globe/geo surfaces,
# mapbox tile styles, ink-colored lines/markers. Generators stay theme-agnostic and use
# the *_LIGHT values below; recolor_figure()/retheme_dict() swap them to their dark
# counterparts alongside the category colors. Every value must be unique across all
# swap pairs (the swap is a plain string mapping and must stay reversible).

GEO_LIGHT = {
    "frame":  "#FEFEFE",   # geo bgcolor around the globe (visually = light surface)
    "land":   "#F0F0F1",
    "water":  "#E6F2FE",
    "stroke": "#27272D",   # country borders / coastlines (near-black on light land)
}
GEO_DARK = {
    "frame":  "#101620",
    "land":   "#232B36",
    "water":  "#0D1725",
    "stroke": "#57606C",
}

FIG_INK_LIGHT = "#26262B"   # near-black accent lines/markers on light ground
FIG_INK_DARK = "#D7DCE3"    # ...become near-white on dark ground

MAPBOX_STYLE_LIGHT = "carto-positron"
MAPBOX_STYLE_DARK = "carto-darkmatter"

_NEUTRAL_LIGHT_TO_DARK = {
    **{GEO_LIGHT[k].upper(): GEO_DARK[k] for k in GEO_LIGHT},
    FIG_INK_LIGHT.upper(): FIG_INK_DARK,
    MAPBOX_STYLE_LIGHT.upper(): MAPBOX_STYLE_DARK,
}
_NEUTRAL_DARK_TO_LIGHT = {v.upper(): GEO_LIGHT[k] for k, v in GEO_DARK.items()}
_NEUTRAL_DARK_TO_LIGHT[FIG_INK_DARK.upper()] = FIG_INK_LIGHT
_NEUTRAL_DARK_TO_LIGHT[MAPBOX_STYLE_DARK.upper()] = MAPBOX_STYLE_LIGHT

# Jewel (light) hex <-> Neon (dark) hex, for centrally re-theming finished figures.
# Includes the figure-neutral pairs above, so one swap handles categories + neutrals.
JEWEL_TO_NEON = {SPECTRUM_LIGHT[c].upper(): SPECTRUM_DARK[c] for c in CATEGORY_ORDER}
NEON_TO_JEWEL = {SPECTRUM_DARK[c].upper(): SPECTRUM_LIGHT[c] for c in CATEGORY_ORDER}
JEWEL_TO_NEON.update(_NEUTRAL_LIGHT_TO_DARK)
NEON_TO_JEWEL.update(_NEUTRAL_DARK_TO_LIGHT)


def _swap_hex(obj, mapping):
    """Recursively replace any hex string in a (figure) dict/list with its mapped value."""
    if isinstance(obj, str):
        return mapping.get(obj.upper(), obj)
    if isinstance(obj, list):
        return [_swap_hex(x, mapping) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_swap_hex(x, mapping) for x in obj)
    if isinstance(obj, dict):
        return {k: _swap_hex(v, mapping) for k, v in obj.items()}
    return obj


def recolor_figure(fig, dark: bool = False):
    """
    Re-theme a finished figure for the active color scheme: switch the registered
    template (nbl_light/nbl_dark) and, in dark mode, swap Jewel category colors for
    Neon throughout the figure. Generators stay theme-agnostic; this is the one hook.

    Works on go.Figure instances; dict-style error placeholders are returned as-is.
    """
    if fig is None or (isinstance(fig, dict) and "data" not in fig):
        return fig
    if not hasattr(fig, "update_layout"):
        return fig
    # Embed the template as an object so plotly.js applies it client-side.
    fig.update_layout(template=get_template(dark))
    if dark:
        fig = go.Figure(_swap_hex(fig.to_dict(), JEWEL_TO_NEON))
    return fig


def retheme_dict(fig_dict: dict, dark: bool = False) -> dict:
    """
    Re-theme an *existing* figure dict (as held by a dcc.Graph) for the target
    color scheme, without regenerating it from data. Normalises category colors to
    Jewel first (so it's reversible), then maps to Neon for dark, and embeds the
    matching template object. Used to flip every plot on a theme toggle, including
    static plots that no filter callback touches.
    """
    if not isinstance(fig_dict, dict) or "data" not in fig_dict:
        return fig_dict
    fig_dict = _swap_hex(fig_dict, NEON_TO_JEWEL)
    if dark:
        fig_dict = _swap_hex(fig_dict, JEWEL_TO_NEON)
    fig_dict.setdefault("layout", {})
    fig_dict["layout"]["template"] = get_template(dark).to_plotly_json()
    return fig_dict


# ------------------------------------------------------------------------------------------------
# Network (dash-cytoscape)
# ------------------------------------------------------------------------------------------------
# The nominations network is rendered with dash-cytoscape (Cytoscape.js), not Plotly,
# so it does its layout + dragging + highlight in the browser. Category colors are
# attached to nodes/edges as CSS *classes* (cat-medicine, ...); the palette lives here
# in the stylesheet, so a theme toggle only swaps the stylesheet (Jewel <-> Neon) while
# the elements stay put — the same "generators are theme-agnostic" philosophy as above.

# Raw category label (as found in df_edges) -> canonical CSS class.
CATEGORY_CLASS = {
    "Medicine": "cat-medicine",
    "Physiology or Medicine": "cat-medicine",
    "Physics": "cat-physics",
    "Chemistry": "cat-chemistry",
    "Economics": "cat-economics",
    "Economic Sciences": "cat-economics",
    "Literature": "cat-literature",
    "Peace": "cat-peace",
}


def network_layout_options(name: str = "cola") -> dict:
    """
    Cytoscape layout config for the nominations network.

    'cola' animates the force simulation but runs it *finite*: it settles for a few
    seconds and stops. (The earlier infinite mode kept simulating every animation
    frame for the life of the page — with hundreds of nodes that pinned a CPU core
    and made pan/zoom/drag stutter.) fcose/cose are run-once force layouts; the
    rest are static.
    """
    opts = {"name": name, "animate": True, "fit": True, "padding": 40}
    if name in ("fcose", "cose"):
        opts.update({"nodeRepulsion": 8000, "idealEdgeLength": 90, "nodeSeparation": 120})
    elif name == "concentric":
        opts.update({"concentric": "function(n){ return n.degree(); }",
                     "levelWidth": "function(){ return 2; }"})
    elif name == "cola":
        opts.update({"infinite": False, "maxSimulationTime": 4000, "fit": False,
                     "edgeLength": 110, "nodeSpacing": 8, "handleDisconnected": True,
                     "randomize": False})
    return opts


def network_stylesheet(dark: bool = False) -> list:
    """Base Cytoscape stylesheet (node/edge defaults + per-category colors) for the theme."""
    spec = spectrum(dark)
    neu = neutrals(dark)
    grey = neu["text_muted"]
    label_color = neu["text_primary"]
    node_border = neu["surface"]
    accent = spec["Medicine"]          # highlight / selected color
    laureate_ring = spec["Literature"]  # gold-ish ring for Nobel laureates

    sheet = [
        {
            "selector": "node",
            "style": {
                "width": "data(size)",
                "height": "data(size)",
                "label": "data(label)",
                "font-size": "9px",
                "font-family": FONT_BODY,
                "color": label_color,
                "text-opacity": 0,               # labels hidden until selected/highlighted
                "text-outline-color": neu["surface"],
                "text-outline-width": 2,
                "background-color": grey,
                "border-width": 1.5,
                "border-color": node_border,
                "transition-property": "opacity, background-color, border-width, border-color",
                "transition-duration": "150ms",
            },
        },
        # Nobel laureates get a ring so they stand out regardless of category.
        {"selector": "node.laureate", "style": {"border-width": 3, "border-color": laureate_ring}},
        {
            "selector": "edge",
            "style": {
                "width": 1.4,
                "line-color": grey,
                "curve-style": "bezier",
                "opacity": 0.5,
                "target-arrow-shape": "triangle",
                "target-arrow-color": grey,
                "arrow-scale": 0.7,
                "transition-property": "opacity, line-color, width",
                "transition-duration": "150ms",
            },
        },
    ]
    # Per-category node + edge colors.
    for cat in CATEGORY_ORDER:
        cls = CATEGORY_CLASS[cat]
        color = spec[cat]
        sheet.append({"selector": f"node.{cls}", "style": {"background-color": color}})
        sheet.append({"selector": f"edge.{cls}",
                      "style": {"line-color": color, "target-arrow-color": color}})
    return sheet


def network_highlight_overlay(node_id: str, neighbour_ids, dark: bool = False) -> list:
    """
    Stylesheet appended after network_stylesheet() to spotlight one node's ego-network:
    dim everything, then light up the tapped node (accent) + its neighbours + their edges.
    """
    spec = spectrum(dark)
    accent = spec["Medicine"]
    neighbour_sel = "".join(f'node[id = "{n}"],' for n in neighbour_ids) or "node.__none__,"
    return [
        {"selector": "node", "style": {"opacity": 0.12, "text-opacity": 0}},
        {"selector": "edge", "style": {"opacity": 0.05}},
        {"selector": neighbour_sel[:-1], "style": {"opacity": 1, "text-opacity": 1}},
        {"selector": f'node[id = "{node_id}"]',
         "style": {"opacity": 1, "text-opacity": 1, "background-color": accent,
                   "border-width": 3, "border-color": accent, "z-index": 99}},
        {"selector": f'edge[source = "{node_id}"], edge[target = "{node_id}"]',
         "style": {"opacity": 1, "width": 3}},
    ]
