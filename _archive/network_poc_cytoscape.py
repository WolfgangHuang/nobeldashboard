"""
================================================================================
PROOF OF CONCEPT: Nomination network with dash-cytoscape (Cytoscape.js)
================================================================================

Why this exists
---------------
Your production network graph (plotdatagenerator.create_network_figure) draws
the graph as Plotly go.Scatter line/marker traces on top of a *pre-computed*
NetworkX layout (calculate_layout). Plotly has no graph model, so it cannot do
in-browser force layouts, node dragging with live edges, or neighbourhood
physics. This PoC renders the SAME real data (df_edges.csv) with dash-cytoscape
instead, to compare the interaction quality directly.

What it demonstrates that Plotly can't
--------------------------------------
  * In-browser force layout (fcose / cose) — no NetworkX layout pre-pass needed
  * Drag a node and watch edges follow
  * Smooth pan / zoom / box-select
  * Click a node -> highlight its ego-network, dim the rest (instant, clientside-ish)
  * Switch layout algorithm live from a dropdown

This is intentionally self-contained: it does NOT import plotdatagenerator
(which pulls in pygraphviz, sklearn, polars and loads every CSV at import time).
It reads df_edges.csv directly with pandas so you can run it in isolation.

Run
---
    pip install dash dash-cytoscape pandas
    python network_poc_cytoscape.py
    # open http://127.0.0.1:8051

Notes
-----
  * Laureate status is NOT in df_edges.csv (it is derived from awarded_prizes in
    df_nominations inside build_network_graph). This PoC therefore colours nodes
    by their ROLE (nominator / nominee / both) and edges by CATEGORY, which is
    enough to judge the look & interaction. Wiring in the real laureate/category
    node colouring is trivial once we port it for real.
  * Default filter keeps the node count sane. Widen the year range / add
    categories in the UI to stress-test performance.
"""

import os
from collections import defaultdict

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_cytoscape as cyto

# fcose / cola / dagre etc. are "extra" layouts and must be registered.
cyto.load_extra_layouts()

# ------------------------------------------------------------------------------
# Brand colours — copied from config.py so the PoC stays standalone.
# (config.py: c_blue/c_red/c_orange/c_purple/c_green/c_yellow, mapped per category)
# ------------------------------------------------------------------------------
C = {
    "physics":    "#26384b",  # c_blue
    "medicine":   "#b50603",  # c_red
    "chemistry":  "#f28118",  # c_orange
    "economics":  "#9f5683",  # c_purple
    "peace":      "#2f754e",  # c_green
    "literature": "#edae49",  # c_yellow
    "grey":       "#999999",  # c_grey
    "brand":      "#27262c",  # c_brand_color_main (black)
    "accent":     "#b50603",  # c_brand_color_acc (red) -> highlight colour
    "bg":         "#fafafa",  # c_plot_background
}

CATEGORY_COLOR = {
    "Physics": C["physics"],
    "Chemistry": C["chemistry"],
    "Physiology or Medicine": C["medicine"],
    "Medicine": C["medicine"],
    "Literature": C["literature"],
    "Peace": C["peace"],
    "Economic Sciences": C["economics"],
    "Economics": C["economics"],
}

# Cytoscape can't read arbitrary hex from data() in a mapper, so we attach a
# per-category CSS class to each edge and define the colour in the stylesheet.
CATEGORY_CLASS = {
    "Physics": "cat-physics",
    "Chemistry": "cat-chemistry",
    "Physiology or Medicine": "cat-medicine",
    "Medicine": "cat-medicine",
    "Literature": "cat-literature",
    "Peace": "cat-peace",
    "Economic Sciences": "cat-economics",
    "Economics": "cat-economics",
}

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DF = pd.read_csv(os.path.join(HERE, "df_edges.csv"), sep=";")

ALL_CATEGORIES = sorted(DF["category"].dropna().unique().tolist())
YEAR_MIN, YEAR_MAX = int(DF["year"].min()), int(DF["year"].max())


def build_elements(categories, year_range, max_nodes=600):
    """Return cytoscape `elements` (nodes + edges) for the current filter.

    Mirrors build_network_graph's role logic: a person who only nominates is a
    'nominator', one who is only nominated is a 'nominee', one who does both is
    'both'. Node size scales with degree; edges are coloured by category.
    """
    lo, hi = year_range
    sub = DF[
        DF["category"].isin(categories)
        & (DF["year"] >= lo)
        & (DF["year"] <= hi)
    ]

    # Role detection (single pass).
    is_nominator, is_nominee = set(), set()
    person_name, person_country, degree = {}, {}, defaultdict(int)
    for r in sub.itertuples(index=False):
        is_nominator.add(r.nominator_id)
        is_nominee.add(r.nominee_id)
        person_name[r.nominator_id] = r.nominator_name
        person_name[r.nominee_id] = r.nominee_name
        person_country[r.nominator_id] = r.nominator_country
        person_country[r.nominee_id] = r.nominee_country
        degree[r.nominator_id] += 1
        degree[r.nominee_id] += 1

    # Cap node count so the demo stays responsive: keep highest-degree people.
    keep = set(sorted(degree, key=degree.get, reverse=True)[:max_nodes])

    nodes = []
    for pid in keep:
        if pid in is_nominator and pid in is_nominee:
            role = "both"
        elif pid in is_nominator:
            role = "nominator"
        else:
            role = "nominee"
        nodes.append({
            "data": {
                "id": str(pid),
                "label": person_name.get(pid, str(pid)),
                "role": role,
                "country": person_country.get(pid, "Unknown"),
                "degree": degree[pid],
                # marker size: 12..48 by degree (sqrt keeps hubs from exploding)
                "size": 12 + min(36, (degree[pid] ** 0.5) * 6),
            },
            "classes": f"role-{role}",
        })

    edges = []
    seen = set()
    for r in sub.itertuples(index=False):
        if r.nominator_id not in keep or r.nominee_id not in keep:
            continue
        key = (r.nominator_id, r.nominee_id, r.category)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "data": {
                "source": str(r.nominator_id),
                "target": str(r.nominee_id),
                "category": r.category,
                "year": int(r.year),
            },
            "classes": CATEGORY_CLASS.get(r.category, "cat-other"),
        })

    return nodes + edges


# ------------------------------------------------------------------------------
# Base stylesheet
# ------------------------------------------------------------------------------
BASE_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "width": "data(size)",
            "height": "data(size)",
            "label": "data(label)",
            "font-size": "9px",
            "font-family": "Rubik, sans-serif",
            "color": C["brand"],
            "text-opacity": 0,                 # labels hidden until zoomed/selected
            "background-color": C["grey"],
            "border-width": 1,
            "border-color": "#ffffff",
            "transition-property": "opacity, background-color, border-width",
            "transition-duration": "150ms",
        },
    },
    # Role-based node colours (mirrors NODE_STYLES in create_network_figure).
    {"selector": "node.role-nominator", "style": {"background-color": "#cfcfcf", "border-color": C["grey"]}},
    {"selector": "node.role-nominee",   "style": {"background-color": C["grey"],  "border-color": "#cfcfcf"}},
    {"selector": "node.role-both",      "style": {"background-color": "#6f6f6f",  "border-color": C["brand"]}},

    {
        "selector": "edge",
        "style": {
            "width": 1.4,
            "line-color": C["grey"],
            "curve-style": "bezier",
            "opacity": 0.55,
            "target-arrow-shape": "triangle",
            "target-arrow-color": C["grey"],
            "arrow-scale": 0.7,
            "transition-property": "opacity, line-color, width",
            "transition-duration": "150ms",
        },
    },
    # Per-category edge colours.
    {"selector": "edge.cat-physics",    "style": {"line-color": C["physics"],    "target-arrow-color": C["physics"]}},
    {"selector": "edge.cat-chemistry",  "style": {"line-color": C["chemistry"],  "target-arrow-color": C["chemistry"]}},
    {"selector": "edge.cat-medicine",   "style": {"line-color": C["medicine"],   "target-arrow-color": C["medicine"]}},
    {"selector": "edge.cat-economics",  "style": {"line-color": C["economics"],  "target-arrow-color": C["economics"]}},
    {"selector": "edge.cat-peace",      "style": {"line-color": C["peace"],      "target-arrow-color": C["peace"]}},
    {"selector": "edge.cat-literature", "style": {"line-color": C["literature"], "target-arrow-color": C["literature"]}},
]

LAYOUTS = {
    "cola (LIVE physics, d3-like)": "cola",
    "fcose (force, run once)": "fcose",
    "cose (force, run once)": "cose",
    "concentric (by degree)": "concentric",
    "circle": "circle",
    "breadthfirst (tree)": "breadthfirst",
}


def layout_options(name):
    opts = {"name": name, "animate": True, "fit": True, "padding": 40}
    if name in ("fcose", "cose"):
        opts.update({"nodeRepulsion": 8000, "idealEdgeLength": 90, "nodeSeparation": 120})
    if name == "concentric":
        opts.update({"concentric": "function(n){ return n.degree(); }", "levelWidth": "function(){ return 2; }"})
    if name == "cola":
        # CONTINUOUS simulation: keeps solving forever, so grabbing a node makes
        # its neighbours drift along smoothly (the D3 "live drag" feel). It never
        # freezes — that is the trade-off for the smooth motion.
        opts.update({
            "infinite": True,
            "fit": False,
            "edgeLength": 110,
            "nodeSpacing": 8,
            "handleDisconnected": True,
            "randomize": False,
        })
    return opts


# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = dash.Dash(__name__)
app.title = "Nobel Network — Cytoscape PoC"

DEFAULT_CATS = ["Physics"]
DEFAULT_YEARS = [1901, 1925]

CONTROL_STYLE = {"display": "flex", "flexDirection": "column", "gap": "14px",
                 "width": "300px", "padding": "18px", "background": "#ffffff",
                 "borderRight": "1px solid #eee", "fontFamily": "Rubik, sans-serif"}

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "Rubik, sans-serif",
           "background": C["bg"]},
    children=[
        # ----- Controls -----
        html.Div(style=CONTROL_STYLE, children=[
            html.H3("Network PoC", style={"margin": "0 0 4px", "color": C["brand"]}),
            html.Div("dash-cytoscape · real df_edges.csv",
                     style={"fontSize": "11px", "color": C["grey"], "marginBottom": "8px"}),

            html.Label("Categories", style={"fontWeight": 600, "fontSize": "13px"}),
            dcc.Dropdown(id="cats", options=ALL_CATEGORIES, value=DEFAULT_CATS,
                         multi=True, clearable=False),

            html.Label("Year range", style={"fontWeight": 600, "fontSize": "13px"}),
            dcc.RangeSlider(id="years", min=YEAR_MIN, max=YEAR_MAX,
                            value=DEFAULT_YEARS, step=1,
                            marks={YEAR_MIN: str(YEAR_MIN), YEAR_MAX: str(YEAR_MAX)},
                            tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("Layout", style={"fontWeight": 600, "fontSize": "13px"}),
            dcc.Dropdown(id="layout", options=[{"label": k, "value": v} for k, v in LAYOUTS.items()],
                         value="cola", clearable=False),
            html.Div("Tip: with 'cola', grab a node and drag — neighbours follow.",
                     style={"fontSize": "11px", "color": C["grey"]}),

            html.Button("Reset highlight", id="reset", n_clicks=0,
                        style={"padding": "8px", "cursor": "pointer", "border": "1px solid #ddd",
                               "background": "#fff", "borderRadius": "6px"}),

            html.Hr(style={"border": "none", "borderTop": "1px solid #eee", "width": "100%"}),
            html.Div(id="info", style={"fontSize": "12px", "lineHeight": "1.5",
                                       "color": C["brand"], "overflowY": "auto"}),
        ]),

        # ----- Graph -----
        html.Div(style={"flex": 1, "position": "relative"}, children=[
            cyto.Cytoscape(
                id="cyto",
                elements=build_elements(DEFAULT_CATS, DEFAULT_YEARS),
                layout=layout_options("cola"),
                stylesheet=BASE_STYLESHEET,
                style={"width": "100%", "height": "100%"},
                minZoom=0.2, maxZoom=3,
                boxSelectionEnabled=True,
                wheelSensitivity=0.2,
            ),
        ]),
    ],
)


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------
@callback(
    Output("cyto", "elements"),
    Output("cyto", "layout"),
    Input("cats", "value"),
    Input("years", "value"),
    Input("layout", "value"),
)
def update_graph(cats, years, layout_name):
    cats = cats or DEFAULT_CATS
    return build_elements(cats, years), layout_options(layout_name)


@callback(
    Output("cyto", "stylesheet"),
    Output("info", "children"),
    Input("cyto", "tapNode"),
    Input("reset", "n_clicks"),
    State("cyto", "elements"),
)
def highlight(tap_node, _reset, elements):
    """Click a node -> dim everything, then light up its ego-network."""
    trigger = dash.callback_context.triggered_id
    if trigger == "reset" or not tap_node:
        return BASE_STYLESHEET, html.Em("Click a node to highlight its nominations.")

    nid = tap_node["data"]["id"]

    # Neighbours + who-nominated-whom, straight from the current elements.
    nominated, nominated_by, neighbours = [], [], set()
    name_by_id = {el["data"]["id"]: el["data"].get("label")
                  for el in elements if "source" not in el["data"]}
    for el in elements:
        d = el["data"]
        if "source" not in d:
            continue
        if d["source"] == nid:
            neighbours.add(d["target"])
            nominated.append((name_by_id.get(d["target"], d["target"]), d["category"], d["year"]))
        elif d["target"] == nid:
            neighbours.add(d["source"])
            nominated_by.append((name_by_id.get(d["source"], d["source"]), d["category"], d["year"]))

    neighbour_selectors = "".join(f'node[id = "{n}"],' for n in neighbours) or "node.__none__,"
    highlight_style = BASE_STYLESHEET + [
        {"selector": "node", "style": {"opacity": 0.15, "text-opacity": 0}},
        {"selector": "edge", "style": {"opacity": 0.07}},
        {"selector": neighbour_selectors[:-1],
         "style": {"opacity": 1, "text-opacity": 1}},
        {"selector": f'node[id = "{nid}"]',
         "style": {"opacity": 1, "text-opacity": 1, "background-color": C["accent"],
                   "border-width": 3, "border-color": C["accent"], "z-index": 99}},
        {"selector": f'edge[source = "{nid}"], edge[target = "{nid}"]',
         "style": {"opacity": 1, "width": 3}},
    ]

    def fmt(rows):
        rows = sorted(rows, key=lambda x: x[2])
        out = [html.Li(f"{n} — {c}, {y}") for n, c, y in rows[:8]]
        if len(rows) > 8:
            out.append(html.Li(f"… and {len(rows) - 8} more"))
        return html.Ul(out, style={"margin": "4px 0", "paddingLeft": "16px"}) if out else html.Div("—")

    info = html.Div([
        html.Div(tap_node["data"]["label"], style={"fontWeight": 700, "fontSize": "14px"}),
        html.Div(f'{tap_node["data"]["country"]} · role: {tap_node["data"]["role"]} · '
                 f'{tap_node["data"]["degree"]} connections',
                 style={"color": C["grey"], "marginBottom": "8px"}),
        html.Div("Nominated:", style={"fontWeight": 600}), fmt(nominated),
        html.Div("Was nominated by:", style={"fontWeight": 600, "marginTop": "6px"}), fmt(nominated_by),
    ])
    return highlight_style, info


if __name__ == "__main__":
    app.run(debug=True, port=8051)
