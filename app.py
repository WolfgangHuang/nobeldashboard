##################################################################################################
# Library Imports
##################################################################################################

# import time
import os
# from dotenv import load_dotenv
import pandas as pd
import polars as pl
import dash_ag_grid as dag
import dash
from dash import dcc, html, Dash, State, callback_context  #, _dash_renderer
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, MATCH, ALL, ClientsideFunction
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_cytoscape as cyto
cyto.load_extra_layouts()  # registers cola / fcose / dagre / etc.
import json
import inspect
import plotdatagenerator as pdg
from config import APP_CONFIG, get_plot_configs #, PlotConfig
import config as cf
import theme as th


##################################################################################################
# Data Loading
##################################################################################################

# Get the path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Load tables {#f9e}
df_laureates = pd.read_csv('df_laureates_enriched_redux_clean.csv', sep=';', encoding="utf8", index_col=0)
df_prizes = pd.read_csv('df_prizes_enriched_redux_clean.csv', sep=';', encoding="utf8", index_col=0)
df_prizestats = pd.read_csv("df_prizestats.csv", sep=';', encoding="utf8")
# df_nominations = pl.read_csv('nominations_full.csv', separator=';', encoding='utf8')
df_nominations = pdg.df_nominations


# {#f9e}
# Generate plot configurations with loaded data
plot_configs = get_plot_configs(df_laureates, df_prizes, df_nominations, pdg.lastyearincluded)

df = pdg.count_per_country()
max_prize_count = df['Count'].max()
lastyearincluded = pdg.get_lastyearincluded()
numberofprizes = df_prizes.shape[0]
totalprizeamount = pdg.generate_var_prizeamount()

standard_loader_message = dmc.Loader(html.Div("Initializing tab..."))




##################################################################################################
# Helper Functions
##################################################################################################

def get_last_update_timestamp():
    """
    Read the last API update timestamp from file.
    
    Retrieves the timestamp of the most recent data update from the Nobel API,
    stored in a text file. Used to display data freshness in the UI.
    
    Returns:
        str: Timestamp string (format varies) or "Never" if no update recorded.
        
    Note:
        Timestamp file path: "last_api_update.txt" in the current directory.
    """
    timestamp_file = "last_api_update.txt"
    if os.path.exists(timestamp_file):
        try:
            with open(timestamp_file, 'r') as f:
                return f.read().strip()
        except Exception:
            return "Never"
    return "Never"



##################################################################################################
# Dashboard Main Setup
##################################################################################################

# _dash_renderer._set_react_version("18.2.0")

app = Dash(
    external_stylesheets=[
        th.GOOGLE_FONTS_URL,
        *APP_CONFIG['external_stylesheets'],
        dmc.styles.ALL
    ],
    title=APP_CONFIG['title'],
    suppress_callback_exceptions=APP_CONFIG['suppress_callback_exceptions']
)

server = app.server


##################################################################################################
# Functions to generate plots in the layout
##################################################################################################

# All functions get their parameters from the PlotConfig instance provided to them (as id) from config.py.
# If you want to change the plot parameters, you need to change the plot_configs dictionary.


# ------------------------------------------------------------------------------------------------
# Loading-Spinners
# ------------------------------------------------------------------------------------------------

#1: Create loading-spinners (placeholders) for the plots while they are being generated {#eef}
def generate_loader_spinner(id):
    """
    Create a loading spinner placeholder for lazy-loaded plots.
    
    Generates a container with a spinner that displays while the actual plot
    is being generated asynchronously. The spinner is replaced with the plot
    content once generation completes.
    
    Args:
        id (str): Unique identifier for the plot. Used to create pattern-matched
            component IDs for the callback system.
            
    Returns:
        html.Div: Container with nested spinner and hidden content area.
        
    Component Structure:
        - outer-container: Main wrapper
        - border: Spinner wrapper (initially visible)
        - spinner: Animated loading indicator
        - inner-container: Plot content area (initially hidden)
    """
    return html.Div(
        id={'type':'outer-container', 'index':id},
        children=[
            html.Div(
                dmc.Loader(
                    id={'type':'spinner', 'index':id},
                    color="economics",  # Mantine theme ramp (auto-adjusts in dark mode)
                    size="md",  # Available sizes: xs, sm, md, lg, xl
                    variant="dots",  # Available variants: oval, dots, bars
                ),
                className="loader-spinner",
                id={'type':'border', 'index':id},
                style={"display": "block"},
            ),
            html.Div(
                id={'type':'inner-container', 'index':id},
                style={"display": "none"},  # Initially hidden
            ),
        ],
    )

# Callback to display the plots once they are generated {#efe, 23}
@app.callback(
    Output({'type': 'inner-container', 'index': MATCH}, "children"),
    Output({'type': 'spinner', 'index': MATCH}, "style"),  # Hide loader
    Output({'type': 'border', 'index': MATCH}, "style"),  # Hide border
    Output({'type': 'inner-container', 'index': MATCH}, "style"),  # Show graph
    Input({'type': 'inner-container', 'index': MATCH}, "id"),  # Trigger on app load
)
def display_plot(triggered_id):
    """
    Callback to render plots and replace loading spinners.
    
    Triggered when a plot container is created, this callback generates
    the actual plot content and swaps out the loading spinner.
    
    Args:
        triggered_id (dict): Pattern-matched component ID containing
            the plot identifier in the 'index' key.
            
    Returns:
        tuple: (plot_layout, spinner_style, border_style, container_style)
            - plot_layout: Generated plot component tree
            - spinner_style: {"display": "none"} to hide spinner
            - border_style: {"display": "none"} to hide border
            - container_style: {"display": "block"} to show plot
            
    Raises:
        ValueError: If plot_id not found in plot_configs.
    """
    # Extract the plot_id from the triggered ID
    plot_id = triggered_id["index"]
    # print("Loader: Plot ID:", plot_id)
    # Look up the corresponding plot configuration
    config = plot_configs.get(plot_id)

    if not config:
        raise ValueError(f"Plot configuration for ID {plot_id} not found.")

    # Generate the layout for the plot
    # fig = generate_plot_in_layout_class(config)
    fig_in_layout = plot_configs[plot_id].generate_layout()
    # Hide the loader and show the graph

    return fig_in_layout, {"display": "none"}, {"display": "none"}, {"display": "block"}


# ------------------------------------------------------------------------------------------------
# Layout for STANDARD plots (based on df_laureates and df_prizes)
# ------------------------------------------------------------------------------------------------

def generate_plot_in_layout_standard(plot_config):
    """
    Generate the complete layout for a standard (non-nominations) plot.
    
    Creates a widget container with header, filter controls, the plot itself,
    and optional footer. Used for all plots based on df_laureates and df_prizes.
    
    Args:
        plot_config (PlotConfig): Configuration object containing:
            - plot_id: Unique identifier
            - header/subheader: Title text
            - plot_generator: Function name to generate the figure
            - plot_generator_kwargs: Arguments for the generator
            - style: CSS styling for the plot
            - footer: Optional footer content
            - badges: Filter state indicators
            
    Returns:
        dmc.SimpleGrid: Complete widget layout with header, plot, and controls.
        
    Raises:
        ValueError: If plot_generator function not found in pdg module.
        PreventUpdate: If plot_category is 'nominations' (wrong function).
    """
    plot_generator = plot_config.get_plot_generator(pdg)
    if not plot_generator:
        raise ValueError(f"Plot generator '{plot_config.plot_generator}' not found.")
    
    # plot_id = plot_id["index"]
    # plot_config = plot_configs.get(plot_id)
    # if not plot_config:
    #     raise ValueError(f"Plot configuration for ID {plot_id} not found.")

    if plot_config.plot_category == "nominations":
        raise PreventUpdate

    figure = plot_generator(**plot_config.plot_generator_kwargs)
    figure = th.recolor_figure(figure, dark=False)  # embed light template on first render
    filter_drawer = cf.render_filter_standard(plot_config)

    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 1},
        spacing="sm",
        verticalSpacing="sm",
        children=[
            html.Div(
                id=plot_config.plot_id,
                children=[
                    # Header section
                    html.Div([
                        html.H3(plot_config.header, className="plot-header"),
                        dmc.Grid(
                            columns=24,
                            children=[
                                dmc.GridCol(html.Div(plot_config.subheader, className="plot-subheader"), 
                                          span={"base": 24, "md": 14}),
                                dmc.GridCol(html.Div(plot_config.generate_badges()), 
                                          span={"base": 24, "md": 7}),
                                dmc.GridCol(
                                    html.Div([
                                        dmc.Button(
                                            "Filter", 
                                            variant="gradient", 
                                            gradient={"from": "economics", "to": "physics"}, 
                                            size="xs", 
                                            id={"type": "filter-button", "index": plot_config.plot_id},
                                        ),
                                        filter_drawer,
                                    ], style={"textAlign": "right"}),
                                    span={"base": 24, "md": 3}
                                ),
                            ],
                        ),
                    ], className="widget-header"),
                    
                    # Plot section
                    html.Div([
                        dcc.Graph(
                            id={"type": "plot", "index": plot_config.plot_id},
                            figure=figure,
                            config={"displayModeBar": False},
                            style=plot_config.style,
                        ),
                    ], className="widget-plot"),
                    
                    # Footer section
                    html.Div(
                        plot_config.footer,
                        className="widget-footer",
                    ) if plot_config.footer else None,
                ],
                className="widget-container",
            ),
        ],
    )

# ------------------------------------------------------------------------------------------------
# Callback: Filter to update STANDARD plots; definitions are in config.py
# ------------------------------------------------------------------------------------------------

@app.callback(
    Output({"type": "plot", "index": MATCH}, "figure"),
    [
        Input({"type": "chip-medicine", "index": MATCH}, "checked"),
        Input({"type": "chip-physics", "index": MATCH}, "checked"),
        Input({"type": "chip-chemistry", "index": MATCH}, "checked"),
        Input({"type": "chip-economics", "index": MATCH}, "checked"),
        Input({"type": "chip-literature", "index": MATCH}, "checked"),
        Input({"type": "chip-peace", "index": MATCH}, "checked"),
        Input({"type": "chip-female", "index": MATCH}, "checked"),
        Input({"type": "chip-male", "index": MATCH}, "checked"),
        Input({"type": "slider-timerange", "index": MATCH}, "value"),
        Input({"type": "dropdown-timerange", "index": MATCH}, "value"),
        Input({"type": "switch-update", "index": MATCH}, "checked"),
        Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
        Input({"type": "custom-filter", "index": MATCH, "filter": ALL}, "value"),
        Input("theme-store", "data")
    ],
    [State({"type": "plot", "index": MATCH}, "id")]
)
def update_plot(
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    chip_female, chip_male, slider_timerange, dropdown_timerange, switch_update, n_clicks,
    custom_filter_values, theme, plot_id
):
    dark = (theme == "dark")
    # Extract plot_id string first
    plot_id_str = plot_id["index"]
    plot_config = plot_configs.get(plot_id_str)
    
    if not plot_config:
        raise ValueError(f"Plot configuration for ID {plot_id_str} not found.")
    
    # CRITICAL: If the plot is of type nominations, do not update here
    if plot_config.plot_category == "nominations":
        raise PreventUpdate
    
    # Ensure the callback is only triggered when the submit button is clicked
    ctx = dash.callback_context

    # A pure theme switch is handled centrally by retheme_all_plots (which flips every
    # plot, including ones this MATCH callback never reaches). Skip it here.
    triggered_props = [t['prop_id'] for t in ctx.triggered]
    if triggered_props and all('theme-store' in p for p in triggered_props):
        raise PreventUpdate

    if switch_update:
        if not any('"type":"submit-button"' in trigger['prop_id'] for trigger in ctx.triggered):
            raise PreventUpdate

    # Extract and update filter states
    selected_categories = pdg.define_category_states(
        chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace
    )
    selected_genders = pdg.define_gender_states(chip_female, chip_male)

    # Combine standard kwargs with filter values
    plot_generator_kwargs = plot_config.plot_generator_kwargs or {}
    plot_generator_kwargs.update({
        "categories": selected_categories,
        "gender": selected_genders,
        "timerange": slider_timerange,
        "timerange_field": dropdown_timerange
    })

    if custom_filter_values:
        # Parse each input id, skipping plain (non-pattern-matched) component ids
        # which aren't valid JSON (e.g. "slider-timerange.value")
        custom_filter_patterns = []
        for key in ctx.inputs.keys():
            try:
                parsed_id = json.loads(key.split('.')[0])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_id, dict) and parsed_id.get("type") == "custom-filter":
                custom_filter_patterns.append(parsed_id)

        # Add custom filter values to kwargs using the filter name from the pattern
        for pattern, value in zip(custom_filter_patterns, custom_filter_values):
            if value is not None:  # Only add non-None values
                filter_name = pattern["filter"]  # This gets "city" from the pattern
                plot_generator_kwargs[filter_name] = value

    # Generate the updated figure
    try:
        plot_generator = plot_config.get_plot_generator(pdg)
        if "dark" in inspect.signature(plot_generator).parameters:
            plot_generator_kwargs["dark"] = dark
        updated_figure = plot_generator(**plot_generator_kwargs)
        updated_figure = th.recolor_figure(updated_figure, dark)
    except Exception as e:
        print(f"Error generating plot for {plot_id_str}: {e}")
        updated_figure = {"data": [], "layout": {"title": "Error generating plot"}}

    return updated_figure


# Callback to toggle the modal for STANDARD plots
@app.callback(
    Output({"type": "filter-modal", "index": MATCH}, "opened"),
    Input({"type": "filter-button", "index": MATCH}, "n_clicks"),
    Input({"type": "close-button", "index": MATCH}, "n_clicks"),
    Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
    State({"type": "filter-modal", "index": MATCH}, "opened"),
    prevent_initial_call=True,
)
def toggle_modal_standard(nc1, nc2, nc3, opened):
    """
    Toggle the filter drawer/modal for standard plots.
    
    Opens or closes the filter panel when any of the trigger buttons
    (filter, close, submit) are clicked.
    
    Args:
        nc1: Filter button click count (unused, triggers callback).
        nc2: Close button click count (unused, triggers callback).
        nc3: Submit button click count (unused, triggers callback).
        opened (bool): Current open/closed state of the modal.
        
    Returns:
        bool: Inverted state (True if was False, False if was True).
    """
    return not opened



# ------------------------------------------------------------------------------------------------
# Layout for NOMINATIONS plot (based on df_nominations)
# ------------------------------------------------------------------------------------------------

def generate_plot_in_layout_nominations(plot_config):
    """
    Generate the complete layout for nominations-based plots.
    
    Creates a widget container specialized for nomination network graphs
    and maps, including click handlers for node interaction and a dcc.Store
    for graph connection data.
    
    Args:
        plot_config (PlotConfig): Configuration object with plot_category='nominations'.
            Must include plot_generator that returns (figure, networkx_graph) tuple.
            
    Returns:
        dmc.SimpleGrid: Complete widget layout including:
            - dcc.Store for graph connections (enables click interactions)
            - Header with title, badges, and filter button
            - Plot with dcc.Loading wrapper
            - Click info display (network graphs only)
            - Reset button (network graphs only)
            - Optional footer
            
    Raises:
        ValueError: If plot_generator function not found.
        
    Note:
        Network graphs return (fig, G) tuple; maps return (fig, None).
    """
    # The nominations NETWORK is rendered with dash-cytoscape, not Plotly.
    if plot_config.plot_generator == "generate_network":
        return generate_network_in_layout_cytoscape(plot_config)

    plot_generator = plot_config.get_plot_generator(pdg)
    if not plot_generator:
        raise ValueError(f"Plot generator '{plot_config.plot_generator}' not found.")

    result = plot_generator(**plot_config.plot_generator_kwargs)

    # Unpack figure and graph
    if isinstance(result, tuple):
        figure, G = result  # Unpack both figure and graph
        # Extract connections for initial storage
        #initial_graph_connections = pdg.extract_graph_connections(G)

        if G is not None:
            initial_graph_connections = pdg.extract_graph_connections(G)
        else:
            initial_graph_connections = None  # Map has no graph

    else:
        figure = result
        initial_graph_connections = None  # Fallback

    filter_drawer = cf.render_filter_nominations(plot_config)

    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 1},
        spacing="sm",
        verticalSpacing="sm",
        children=[
            dcc.Store(
                id={"type": "nom-graph-store", "index": plot_config.plot_id},
                data=initial_graph_connections  # <-- WICHTIG: Initiale Daten setzen
            ),
            html.Div(
                id=plot_config.plot_id,
                children=[
                    # Header section
                    html.Div([
                        html.H3(plot_config.header, className="plot-header"),
                        dmc.Grid(
                            columns=24,
                            children=[
                                dmc.GridCol(html.Div(plot_config.subheader, className="plot-subheader"), 
                                          span={"base": 24, "md": 14}),
                                dmc.GridCol(html.Div(plot_config.generate_badges()), 
                                          span={"base": 24, "md": 7}),
                                dmc.GridCol(
                                    html.Div([
                                        dmc.Button(
                                            "Filter", 
                                            variant="gradient", 
                                            gradient={"from": "economics", "to": "physics"}, 
                                            size="xs", 
                                            id={"type": "nom-filter-button", "index": plot_config.plot_id},
                                        ),
                                        filter_drawer,
                                    ], style={"textAlign": "right"}),
                                    span={"base": 24, "md": 3}
                                ),
                            ],
                        ),
                    ], className="widget-header"),
                    
                    # Plot section WITH click-info for nominations
                    dcc.Loading(
                        id={"type": "nom-loading", "index": plot_config.plot_id},
                        type="circle",
                        color=th.SPECTRUM_LIGHT["Economics"],  # brand accent (readable in both modes)
                        children=[
                            html.Div([
                                dcc.Graph(
                                    id={"type": "nom-plot", "index": plot_config.plot_id},
                                    figure=figure,
                                    config={"displayModeBar": False},
                                    style=plot_config.style,
                                ),
                            ], className="widget-plot"),
                        ]
                    ),
                    # Click info display (only for network graph, not map)
                    dcc.Markdown(
                        id={"type": "nom-click-info", "index": plot_config.plot_id},
                        children="Click on any person to highlight their nominations",
                        style={'textAlign': 'left', 'padding': '2px', 'fontSize': '12px'}
                    ) if plot_config.plot_generator == "generate_network" else None,
                    # Reset button (only for network graph, not map)
                    dmc.Button(
                        "Reset View",
                        id={"type": "nom-reset-button", "index": plot_config.plot_id}, 
                        size="xs",
                        variant="outline",
                        style={'marginBottom': '10px'}
                    ) if plot_config.plot_generator == "generate_network" else None,
                    # Footer section
                    html.Div(
                        plot_config.footer,
                        className="widget-footer",
                    ) if plot_config.footer else None,
                ],
                className="widget-container",
            ),
        ],
    )


# ------------------------------------------------------------------------------------------------
# NOMINATIONS network — dash-cytoscape (replaces the Plotly network graph)
# ------------------------------------------------------------------------------------------------

NETWORK_DEFAULT_INFO = "Click on any person to highlight their nominations"

# Single network instance -> concrete component ids (no MATCH needed). The filter
# controls inside the drawer keep their pattern ids with this concrete index.
NETWORK_PLOT_ID = "fig_network_nominations"


def generate_network_in_layout_cytoscape(plot_config):
    """
    Build the nominations-network widget backed by dash-cytoscape.

    Mirrors generate_plot_in_layout_nominations' header + filter drawer, but the plot
    itself is a cyto.Cytoscape (in-browser force layout, node dragging, click-to-
    highlight) instead of a dcc.Graph. Category colors/laureate rings come from the
    theme stylesheet; the ego-network highlight + info text are driven by the callback
    below using the same graph-connections store as the old Plotly path.
    """
    elements, G = pdg.generate_network_elements(**plot_config.plot_generator_kwargs)
    initial_connections = pdg.extract_graph_connections(G) if G is not None else None

    filter_drawer = cf.render_filter_nominations(plot_config)

    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 1},
        spacing="sm",
        verticalSpacing="sm",
        children=[
            dcc.Store(id="nom-cyto-graph-store", data=initial_connections),
            dcc.Store(id="nom-cyto-fit-store"),
            html.Div(
                id=plot_config.plot_id,
                children=[
                    # Header section
                    html.Div([
                        html.H3(plot_config.header, className="plot-header"),
                        dmc.Grid(
                            columns=24,
                            children=[
                                dmc.GridCol(html.Div(plot_config.subheader, className="plot-subheader"),
                                            span={"base": 24, "md": 14}),
                                dmc.GridCol(html.Div(plot_config.generate_badges()),
                                            span={"base": 24, "md": 7}),
                                dmc.GridCol(
                                    html.Div([
                                        dmc.Button(
                                            "Filter",
                                            variant="gradient",
                                            gradient={"from": "economics", "to": "physics"},
                                            size="xs",
                                            id={"type": "nom-filter-button", "index": plot_config.plot_id},
                                        ),
                                        filter_drawer,
                                    ], style={"textAlign": "right"}),
                                    span={"base": 24, "md": 3}
                                ),
                            ],
                        ),
                    ], className="widget-header"),

                    # Cytoscape network
                    html.Div(
                        cyto.Cytoscape(
                            id="nom-cyto",
                            elements=elements,
                            layout=th.network_layout_options("cola"),
                            stylesheet=th.network_stylesheet(dark=False),
                            style={"width": "100%", "height": "640px"},
                            minZoom=0.2, maxZoom=3,
                            boxSelectionEnabled=True,
                            wheelSensitivity=0.2,
                        ),
                        className="widget-plot",
                    ),

                    # Click info + reset
                    dcc.Markdown(
                        id="nom-cyto-click-info",
                        children=NETWORK_DEFAULT_INFO,
                        style={'textAlign': 'left', 'padding': '2px', 'fontSize': '12px'},
                    ),
                    dmc.Button(
                        "Reset View",
                        id="nom-cyto-reset-button",
                        size="xs",
                        variant="outline",
                        style={'marginBottom': '10px'},
                    ),

                    # Footer section
                    html.Div(
                        plot_config.footer,
                        className="widget-footer",
                    ) if plot_config.footer else None,
                ],
                className="widget-container",
            ),
        ],
    )


@app.callback(
    Output("nom-cyto", "elements"),
    Output("nom-cyto", "stylesheet"),
    Output("nom-cyto", "layout"),
    Output("nom-cyto-graph-store", "data"),
    Output("nom-cyto-click-info", "children"),
    [
        Input("nom-cyto-reset-button", "n_clicks"),
        Input("nom-cyto", "tapNodeData"),
        Input({"type": "nom-submit-button", "index": NETWORK_PLOT_ID}, "n_clicks"),
        Input({"type": "nom-switch-update", "index": NETWORK_PLOT_ID}, "checked"),
        Input({"type": "nom-algorithm", "index": NETWORK_PLOT_ID}, "value"),
        Input("theme-store", "data"),
    ],
    [
        # Category filters
        State({"type": "nom-chip-medicine", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-physics", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-chemistry", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-economics", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-literature", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-peace", "index": NETWORK_PLOT_ID}, "checked"),
        # Timerange
        State({"type": "nom-slider-timerange", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-dropdown-timerange", "index": NETWORK_PLOT_ID}, "value"),
        # Nominator filters
        State({"type": "nom-chip-nominator-female", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-nominator-male", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-nominator-country", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominator-name-input", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominator-searchmode", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominator-laureate-checkbox", "index": NETWORK_PLOT_ID}, "checked"),
        # Nominee filters
        State({"type": "nom-chip-nominee-female", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-chip-nominee-male", "index": NETWORK_PLOT_ID}, "checked"),
        State({"type": "nom-nominee-country", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominee-name-input", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominee-searchmode", "index": NETWORK_PLOT_ID}, "value"),
        State({"type": "nom-nominee-laureate-checkbox", "index": NETWORK_PLOT_ID}, "checked"),
        # Graph store
        State("nom-cyto-graph-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_network_cytoscape(
    reset_clicks, tap_node, submit_clicks, switch_update, layout_name, theme,
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    slider_timerange, dropdown_timerange,
    chip_nominator_female, chip_nominator_male, nominator_country, nominator_name, nominator_searchmode, nominator_islaureate,
    chip_nominee_female, chip_nominee_male, nominee_country, nominee_name, nominee_searchmode, nominee_islaureate,
    graph_connections,
):
    """
    Drive the Cytoscape nominations network: filter recompute, click-to-highlight,
    reset, live layout switch, and dark/light restyle — one callback, dispatched by
    which input fired (mirrors the old Plotly combined callback's structure).
    """
    trigger = dash.callback_context.triggered_id
    dark = (theme == "dark")
    base_style = th.network_stylesheet(dark)
    layout_name = layout_name or "cola"

    # ---- Theme toggle: restyle only ----
    if trigger == "theme-store":
        return dash.no_update, base_style, dash.no_update, dash.no_update, dash.no_update

    # ---- Layout dropdown: re-run layout only ----
    if isinstance(trigger, dict) and trigger.get("type") == "nom-algorithm":
        return dash.no_update, dash.no_update, th.network_layout_options(layout_name), dash.no_update, dash.no_update

    # ---- Reset highlight ----
    if trigger == "nom-cyto-reset-button":
        return dash.no_update, base_style, dash.no_update, dash.no_update, NETWORK_DEFAULT_INFO

    # ---- Node tap: highlight ego-network ----
    if trigger == "nom-cyto":
        if not tap_node:
            raise PreventUpdate
        nid = str(tap_node["id"])
        neighbours = set()
        if graph_connections and nid in graph_connections:
            neighbours.update(s["id"] for s in graph_connections[nid]["successors"])
            neighbours.update(p["id"] for p in graph_connections[nid]["predecessors"])
        style = base_style + th.network_highlight_overlay(nid, neighbours, dark)
        info = pdg.get_person_info_text(graph_connections, nid)
        return dash.no_update, style, dash.no_update, dash.no_update, info

    # ---- Filter submit / live-mode toggle: full recompute ----
    if isinstance(trigger, dict) and trigger.get("type") in ("nom-submit-button", "nom-switch-update"):
        # "Update only on submit" is on: ignore everything except the submit button.
        if switch_update and trigger.get("type") != "nom-submit-button":
            raise PreventUpdate

        # Defaults for any None states.
        chip_medicine = chip_medicine if chip_medicine is not None else True
        chip_physics = chip_physics if chip_physics is not None else True
        chip_chemistry = chip_chemistry if chip_chemistry is not None else True
        chip_economics = chip_economics if chip_economics is not None else True
        chip_literature = chip_literature if chip_literature is not None else True
        chip_peace = chip_peace if chip_peace is not None else True
        slider_timerange = slider_timerange if slider_timerange is not None else [1901, lastyearincluded - 49]
        dropdown_timerange = dropdown_timerange if dropdown_timerange is not None else "nomination"
        chip_nominator_female = chip_nominator_female if chip_nominator_female is not None else True
        chip_nominator_male = chip_nominator_male if chip_nominator_male is not None else True
        chip_nominee_female = chip_nominee_female if chip_nominee_female is not None else True
        chip_nominee_male = chip_nominee_male if chip_nominee_male is not None else True

        selected_categories = pdg.define_category_states(
            chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace
        )
        nominator_gender = pdg.define_gender_states(chip_nominator_female, chip_nominator_male)
        nominee_gender = pdg.define_gender_states(chip_nominee_female, chip_nominee_male)

        kwargs = {
            "data": df_nominations,
            "categories": selected_categories,
            "timerange_nomination": slider_timerange,
            "timerange_field": dropdown_timerange,
            "nominator_gender": nominator_gender,
            "nominee_gender": nominee_gender,
            "nominator_country": nominator_country if nominator_country else "all",
            "nominee_country": nominee_country if nominee_country else "all",
            "nominator_name": nominator_name if nominator_name else "",
            "nominator_search_mode": nominator_searchmode if nominator_searchmode else "all",
            "nominee_name": nominee_name if nominee_name else "",
            "nominee_search_mode": nominee_searchmode if nominee_searchmode else "all",
            "nominator_islaureate": bool(nominator_islaureate),
            "nominee_islaureate": bool(nominee_islaureate),
        }

        try:
            elements, G = pdg.generate_network_elements(**kwargs)
            connections = pdg.extract_graph_connections(G) if G is not None else None
            return elements, base_style, th.network_layout_options(layout_name), connections, NETWORK_DEFAULT_INFO
        except Exception as e:
            print(f"Error generating cytoscape network: {e}")
            import traceback
            traceback.print_exc()
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Error generating network"

    raise PreventUpdate


# The "cola" layout runs with fit=False (see network_layout_options) so the continuous
# force simulation doesn't re-center the viewport on every tick while dragging a node.
# That means nothing ever fits the view to the graph on its own, so the network opens
# zoomed into whatever the previous pan/zoom happened to be (nodes cut off at the edges).
# This clientside callback calls cy.fit() once, shortly after elements/layout change
# (initial load, filter submit, layout switch, reset), without touching it during drag.
app.clientside_callback(
    ClientsideFunction(namespace="network", function_name="fitCytoscape"),
    Output("nom-cyto-fit-store", "data"),
    Input("nom-cyto", "elements"),
    Input("nom-cyto", "layout"),
    Input("nom-cyto-reset-button", "n_clicks"),
)


# ------------------------------------------------------------------------------------------------
# Layout for NOMINATIONS plot (based on df_nominations)
# ------------------------------------------------------------------------------------------------


# Callback to toggle the modal for NOMINATIONS plots
@app.callback(
    Output({"type": "nom-filter-modal", "index": MATCH}, "opened"),
    Input({"type": "nom-filter-button", "index": MATCH}, "n_clicks"),
    Input({"type": "nom-close-button", "index": MATCH}, "n_clicks"),
    Input({"type": "nom-submit-button", "index": MATCH}, "n_clicks"),
    State({"type": "nom-filter-modal", "index": MATCH}, "opened"),
    prevent_initial_call=True,
)
def toggle_modal_nominations(nc1, nc2, nc3, opened):
    return not opened



# Callback to show extra dropdowns in the filter for NOMINATIONS
@app.callback(
    Output({"type": "nom-sfdp-params-container", "index": MATCH}, "style"),
    Input({"type": "nom-algorithm", "index": MATCH}, "value")
)
def toggle_sfdp_params_nominations(algorithm):
    if algorithm == 'graphviz_sfdp':
        return {"display": "block", "margin-top": "10px"}
    else:
        return {"display": "none"}
    


# Combined callback for NOMINATIONS plots (Filter + Click-Handling)
@app.callback(
    [Output({"type": "nom-plot", "index": MATCH}, 'figure'),
     Output({"type": "nom-click-info", "index": MATCH}, 'children'),
     Output({"type": "nom-graph-store", "index": MATCH}, 'data')],  
    [
        # Click & Reset Inputs
        Input({"type": "nom-plot", "index": MATCH}, 'clickData'),
        Input({"type": "nom-reset-button", "index": MATCH}, 'n_clicks'),
        # Filter Inputs
        Input({"type": "nom-submit-button", "index": MATCH}, "n_clicks"),
        Input({"type": "nom-switch-update", "index": MATCH}, "checked"),
    ],
    [
        # Category filters
        State({"type": "nom-chip-medicine", "index": MATCH}, "checked"),
        State({"type": "nom-chip-physics", "index": MATCH}, "checked"),
        State({"type": "nom-chip-chemistry", "index": MATCH}, "checked"),
        State({"type": "nom-chip-economics", "index": MATCH}, "checked"),
        State({"type": "nom-chip-literature", "index": MATCH}, "checked"),
        State({"type": "nom-chip-peace", "index": MATCH}, "checked"),
        
        # Timerange filters
        State({"type": "nom-slider-timerange", "index": MATCH}, "value"),
        State({"type": "nom-dropdown-timerange", "index": MATCH}, "value"),

        # Nominator-specific filters
        State({"type": "nom-chip-nominator-female", "index": MATCH}, "checked"),
        State({"type": "nom-chip-nominator-male", "index": MATCH}, "checked"),
        State({"type": "nom-nominator-country", "index": MATCH}, "value"),
        State({"type": "nom-nominator-name-input", "index": MATCH}, "value"),
        State({"type": "nom-nominator-searchmode", "index": MATCH}, "value"),
        State({"type": "nom-nominator-laureate-checkbox", "index": MATCH}, "checked"),
        
        # Nominee-specific filters
        State({"type": "nom-chip-nominee-female", "index": MATCH}, "checked"),
        State({"type": "nom-chip-nominee-male", "index": MATCH}, "checked"),
        State({"type": "nom-nominee-country", "index": MATCH}, "value"),
        State({"type": "nom-nominee-name-input", "index": MATCH}, "value"),
        State({"type": "nom-nominee-searchmode", "index": MATCH}, "value"),
        State({"type": "nom-nominee-laureate-checkbox", "index": MATCH}, "checked"),
        
        # Algorithm settings - CHANGED TO ALL
        State({"type": "nom-algorithm", "index": MATCH}, "value"),
        State({"type": "nom-sfdp-k-value", "index": MATCH}, "value"),
        State({"type": "nom-sfdp-rf-value", "index": MATCH}, "value"),
        State({"type": "nom-sfdp-overlap", "index": MATCH}, "value"),
        
        # Current figure, Graph Store and Plot ID
        State({"type": "nom-plot", "index": MATCH}, "figure"),
        State({"type": "nom-graph-store", "index": MATCH}, "data"),
        State({"type": "nom-plot", "index": MATCH}, "id"),
    ],
    prevent_initial_call=True,
)
def update_nominations_plot_combined(
    clickData, reset_clicks, submit_clicks, switch_update,
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    slider_timerange, dropdown_timerange,
    chip_nominator_female, chip_nominator_male, nominator_country, nominator_name, nominator_searchmode, nominator_islaureate,
    chip_nominee_female, chip_nominee_male, nominee_country, nominee_name, nominee_searchmode, nominee_islaureate,
    algorithm, sfdp_k_value, sfdp_rf_value, sfdp_overlap,
    current_figure,
    graph_connections,  
    plot_id
):
    """
    Combined callback handling both filter updates AND click interactions for nominations plots.
    Filter updates: Full graph recalculation (slow) + store graph connections
    Click/Reset: Only visual update (fast) using stored graph connections
    """
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    # Extract plot_id
    plot_id_str = plot_id["index"]
    plot_config = plot_configs.get(plot_id_str)
    
    # ========================================================================
    # EXTRACT ALGORITHM VALUES (with fallback for maps without these filters)
    # ========================================================================
    
    # Get first value from list if exists, otherwise use default
    # algorithm = algorithm_list[0] if algorithm_list and len(algorithm_list) > 0 else "graphviz_sfdp"
    # sfdp_k_value = sfdp_k_value_list[0] if sfdp_k_value_list and len(sfdp_k_value_list) > 0 else 1.0
    # sfdp_rf_value = sfdp_rf_value_list[0] if sfdp_rf_value_list and len(sfdp_rf_value_list) > 0 else 1.0
    # sfdp_overlap = sfdp_overlap_list[0] if sfdp_overlap_list and len(sfdp_overlap_list) > 0 else "prism"

    # Use values directly with fallbacks for None
    algorithm = algorithm if algorithm is not None else "graphviz_sfdp"
    sfdp_k_value = sfdp_k_value if sfdp_k_value is not None else 1.0
    sfdp_rf_value = sfdp_rf_value if sfdp_rf_value is not None else 1.0
    sfdp_overlap = sfdp_overlap if sfdp_overlap is not None else "prism"
    
    # ========================================================================
    # FAST PATH: Click or Reset (only visual update, no recalculation)
    # ========================================================================
    
    if 'nom-reset-button' in trigger_id:
        # print("Reset button - fast visual update only")
        updated_fig = pdg.update_network_highlighting(
            current_figure, 
            highlighted_person_id=None, 
            graph_connections=graph_connections
        )
        return updated_fig, "View reset. Click on any person to highlight their nominations", dash.no_update
    
    elif 'clickData' in trigger_id:
        if clickData is None:
            raise PreventUpdate
        
        try:
            # Check if customdata exists
            if 'customdata' not in clickData['points'][0]:
                # print("Click was not on a node with customdata, ignoring")
                raise PreventUpdate
        
            person_id = clickData['points'][0]['customdata']
            # print(f"Click on person {person_id} - fast visual update only")
            
            # Fast highlighting - only change colors/opacity
            updated_fig = pdg.update_network_highlighting(
                current_figure, 
                highlighted_person_id=person_id, 
                graph_connections=graph_connections
            )
            
            # Generate detailed info text
            info_text = pdg.get_person_info_text(graph_connections, person_id)
            
            return updated_fig, info_text, dash.no_update
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate
    
    # ========================================================================
    # SLOW PATH: Filter update (full graph recalculation)
    # ========================================================================
    
    elif 'nom-submit-button' in trigger_id or ('nom-switch-update' in trigger_id and not switch_update):
        # print("Filter update - full graph recalculation")
        
        # Check if switch is on and trigger is not submit button
        if switch_update and 'nom-submit-button' not in trigger_id:
            raise PreventUpdate
        
        # Set default values for None states
        chip_medicine = chip_medicine if chip_medicine is not None else True
        chip_physics = chip_physics if chip_physics is not None else True
        chip_chemistry = chip_chemistry if chip_chemistry is not None else True
        chip_economics = chip_economics if chip_economics is not None else True
        chip_literature = chip_literature if chip_literature is not None else True
        chip_peace = chip_peace if chip_peace is not None else True
        slider_timerange = slider_timerange if slider_timerange is not None else [1901, lastyearincluded-49]
        dropdown_timerange = dropdown_timerange if dropdown_timerange is not None else "nomination"
        chip_nominator_female = chip_nominator_female if chip_nominator_female is not None else True
        chip_nominator_male = chip_nominator_male if chip_nominator_male is not None else True
        chip_nominee_female = chip_nominee_female if chip_nominee_female is not None else True
        chip_nominee_male = chip_nominee_male if chip_nominee_male is not None else True
        nominator_country = nominator_country if nominator_country is not None else []
        nominee_country = nominee_country if nominee_country is not None else []
        nominator_name = nominator_name if nominator_name is not None else ""
        nominator_searchmode = nominator_searchmode if nominator_searchmode is not None else "any"
        nominee_name = nominee_name if nominee_name is not None else ""
        nominee_searchmode = nominee_searchmode if nominee_searchmode is not None else "any"
        nominator_islaureate = nominator_islaureate if nominator_islaureate is not None else False
        nominee_islaureate = nominee_islaureate if nominee_islaureate is not None else False
        # algorithm values already extracted above with fallbacks
        
        # Prepare kwargs with current filter settings
        selected_categories = pdg.define_category_states(
            chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace
        )
        nominator_gender = pdg.define_gender_states(chip_nominator_female, chip_nominator_male)
        nominee_gender = pdg.define_gender_states(chip_nominee_female, chip_nominee_male)
        
        plot_generator_kwargs = plot_config.plot_generator_kwargs or {}
        plot_generator_kwargs.update({
            "categories": selected_categories,
            "timerange_nomination": slider_timerange,
            "timerange_field": dropdown_timerange,
            "nominator_gender": nominator_gender,
            "nominee_gender": nominee_gender,
            "nominator_country": nominator_country,
            "nominee_country": nominee_country,
            "nominator_name": nominator_name,
            "nominator_search_mode": nominator_searchmode,
            "nominee_name": nominee_name,
            "nominee_search_mode": nominee_searchmode,
            "nominator_islaureate": nominator_islaureate,
            "nominee_islaureate": nominee_islaureate,
            "algorithm": algorithm,
            "sfdp_k_value": sfdp_k_value,
            "sfdp_rf_value": sfdp_rf_value,
            "sfdp_overlap": sfdp_overlap
        })
        
        # print("Nominations Updater: kwargs:", plot_generator_kwargs)
        
        try:
            # Full recalculation - slow but necessary for filter changes
            plot_generator = plot_config.get_plot_generator(pdg)
            updated_figure, G = plot_generator(**plot_generator_kwargs)
            
            # Extract connections for storage
            #new_graph_connections = pdg.extract_graph_connections(G)

                # Extract graph connections only if G exists (network plots have graphs, maps don't)
            if G is not None:
                new_graph_connections = pdg.extract_graph_connections(G)
            else:
                new_graph_connections = None  # Map has no graph
                    
            return updated_figure, "Filter applied. Click on any person to highlight their nominations", new_graph_connections
            
        except Exception as e:
            print(f"Error generating nominations plot for {plot_id_str}: {e}")
            import traceback
            traceback.print_exc()
            updated_figure = {"data": [], "layout": {"title": "Error generating plot"}}
            return updated_figure, "Error generating plot", dash.no_update
    
    raise PreventUpdate


# Callback to calculate selected nominations in the filter (live count)
@app.callback(
    Output({"type": "nom-nomination-count", "index": MATCH}, "children"),
    [
        Input({"type": "nom-chip-medicine", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-physics", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-chemistry", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-economics", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-literature", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-peace", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-nominator-female", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-nominator-male", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-nominee-female", "index": MATCH}, "checked"),
        Input({"type": "nom-chip-nominee-male", "index": MATCH}, "checked"),
        Input({"type": "nom-nominator-country", "index": MATCH}, "value"),
        Input({"type": "nom-nominee-country", "index": MATCH}, "value"),
        Input({"type": "nom-slider-timerange", "index": MATCH}, "value"),
        Input({"type": "nom-dropdown-timerange", "index": MATCH}, "value"),
        Input({"type": "nom-nominator-name-input", "index": MATCH}, "value"),
        Input({"type": "nom-nominator-searchmode", "index": MATCH}, "value"),
        Input({"type": "nom-nominee-name-input", "index": MATCH}, "value"),
        Input({"type": "nom-nominee-searchmode", "index": MATCH}, "value"),
        Input({"type": "nom-nominator-laureate-checkbox", "index": MATCH}, "checked"),
        Input({"type": "nom-nominee-laureate-checkbox", "index": MATCH}, "checked"),

    ],
    prevent_initial_call=False  # Also runs on initial load
)
def update_nomination_count_live(
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    chip_nominator_female, chip_nominator_male,
    chip_nominee_female, chip_nominee_male,
    nominator_country, nominee_country,
    timerange, timerange_type,
    nominator_name, nominator_searchmode,
    nominee_name, nominee_searchmode,
    nominator_islaureate, nominee_islaureate
):
    """
    Live update of nomination count based on current filter settings.
    Uses pre-computed df_edges from plotdatagenerator for performance.
    Shows both total count and count with map coordinates.
    """
   
    # Extract and update filter states
    selected_categories = pdg.define_category_states(
        chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace
    )
   
    nominator_gender = pdg.define_gender_states(chip_nominator_female, chip_nominator_male)
    nominee_gender = pdg.define_gender_states(chip_nominee_female, chip_nominee_male)
    
    # Use pre-computed df_edges from plotdatagenerator (loaded at startup)
    # instead of calling transform_to_edges() which iterates over all nominations
    
    # Call filter function
    df_filtered_edges = pdg.filter_edges(
        data=pdg.df_edges,
        categories=selected_categories,
        nominator_gender=nominator_gender,
        nominee_gender=nominee_gender,
        nominator_country=nominator_country if nominator_country else [],
        nominee_country=nominee_country if nominee_country else [],
        timerange_nomination=timerange,
        timerange_field=timerange_type,
        nominator_name=nominator_name if nominator_name else [],
        nominator_search_mode=nominator_searchmode,
        nominee_name=nominee_name if nominee_name else [],
        nominee_search_mode=nominee_searchmode,
        nominator_islaureate=nominator_islaureate,
        nominee_islaureate=nominee_islaureate
    )
   
    # Get total count
    n_nominations = len(df_filtered_edges)
    
    # Count how many have valid coordinates for map display
    n_with_coords = len(df_filtered_edges.filter(
        pl.col("nominator_lat").is_not_null() &
        pl.col("nominator_lon").is_not_null() &
        pl.col("nominee_lat").is_not_null() &
        pl.col("nominee_lon").is_not_null()
    ))
   
    # Format output - show both counts
    if n_with_coords == n_nominations:
        return f"{n_nominations:,} nominations selected"
    else:
        return f"{n_nominations:,} nominations selected ({n_with_coords:,} with map coordinates)"



# ------------------------------------------------------------------------------------------------
# Layout for PNGs (instead of plotly figures)
# ------------------------------------------------------------------------------------------------

def generate_png_in_layout(
    cols= {"base": 1, "sm": 1},
    header= "Generic Plot Title", 
    subheader="", 
    datafrom="1901", 
    datato=lastyearincluded, 
    badges=[dmc.Badge("All Categories", variant="outline", color="economics")],
    code="", 
    filepath="",
    style={'width': '80vw', 'height': '50vh'}, 
    footer=""
):
    return dmc.SimpleGrid(
        cols=cols,
        spacing="sm",
        verticalSpacing="sm",
        children=[
            html.Div(
                [
                    # Header
                    html.Div(
                        [
                            html.H3(header, className="plot-header"),
                            html.P(subheader, className="plot-subheader") if subheader else None,
                            dmc.Group(
                                [
                                    dmc.Badge(f"{datafrom} - {datato}", variant="outline", color="economics"),
                                    *badges,
                                ]
                            ),
                        ],
                        className="widget-title",
                    ),
                    # Selection area
                    html.Div(code) if code else None,  # Include code only if provided
                    # Content
                    html.Div(
                        dcc.Loading(
                            html.Img(src=filepath, style=style),
                        ),
                        className="widget-content",
                    ),
                    # Footer
                    html.Div(
                        [
                            *footer,
                        ],
                        className="widget-footer",
                    ) if footer else None,
                ],
                className="widget-container",
            )
        ]
    )


##################################################################################################
##################################################################################################
##################################################################################################
# Sections
##################################################################################################
##################################################################################################
##################################################################################################



##################################################################################################
# Section Overview
##################################################################################################

# Definition of the AG Grid to display the prize stats table
ag_df_prizestats = dag.AgGrid(
    id="prizestats-aggrid",
    rowData=df_prizestats.to_dict("records"),
    #columnDefs=[{"field": i} for i in df_prizestats.columns],
    columnDefs = [
        {'field': 'Category', 'width': 100, 'suppressSizeToFit': True},
        {'field': '1/1', 'width': 50,},
        {'field': '1/2', 'width': 50,},
        {'field': '1/3', 'width': 50},
        {'field': 'Organisation', 'width': 70},
        {'field': 'Multiple Recipients', 'width': 150},
        {'field': 'Posthumous', 'width': 100},
        {'field': 'Declined', 'width': 100},
        {'field': 'Forced To Decline', 'width': 100},
        {'field': 'Not Awarded In', 'width': 150},
    ],
    columnSize="sizeToFit",
    dashGridOptions={
        "pagination": False,
    }
)

# Callback that triggers the initial loading of the Content for the tab Overview
##################################################################################################

def spektrum_verteilung(df_prizes_filtered, dark=False):
    """
    Build the 'Spektrum-Verteilung' card body: one horizontal bar per prize category
    (label + category-colored track fill + mono count), ordered by count. Pure CSS/HTML
    so it re-themes on toggle via update_overview_content re-render.
    """
    counts = (
        df_prizes_filtered["Prize0_Category"]
        .replace({"Economic Sciences": "Economics"})
        .value_counts()
    )
    ordered = counts.reindex(th.CATEGORY_ORDER).fillna(0).astype(int).sort_values(ascending=False)
    spec = th.spectrum(dark)
    max_c = int(ordered.max()) or 1
    rows = []
    for cat, cnt in ordered.items():
        rows.append(
            html.Div(
                [
                    html.Span(cat, className="spektrum-label"),
                    html.Div(
                        html.Div(
                            className="spektrum-fill",
                            style={"width": f"{cnt / max_c * 100:.1f}%", "backgroundColor": spec[cat]},
                        ),
                        className="spektrum-track",
                    ),
                    html.Span(f"{cnt}", className="spektrum-count"),
                ],
                className="spektrum-row",
            )
        )
    return html.Div(rows, className="spektrum-list")


def render_overview_content():
    return dmc.Stack(
        id="section-overview",
        children=[
            # Compact filter toolbar (single wrapping row): category chips · gender · year range
            html.Div(
                dmc.Group(
                    [
                        # Category toggle chips
                        html.Div(
                            [
                                html.Div("CATEGORIES", className="toolbar-eyebrow"),
                                dmc.Group(
                                    [
                                        dmc.Chip("Medicine", checked=True, color="medicine", id="chip-medicine"),
                                        dmc.Chip("Physics", checked=True, color="physics", id="chip-physics"),
                                        dmc.Chip("Chemistry", checked=True, color="chemistry", id="chip-chemistry"),
                                        dmc.Chip("Economics", checked=True, color="economics", id="chip-economics"),
                                        dmc.Chip("Literature", checked=True, color="literature", id="chip-literature"),
                                        dmc.Chip("Peace", checked=True, color="peace", id="chip-peace"),
                                    ],
                                    gap="xs",
                                ),
                            ]
                        ),

                        html.Div(className="toolbar-divider"),

                        # Gender segmented control
                        html.Div(
                            [
                                html.Div("GENDER", className="toolbar-eyebrow"),
                                dmc.SegmentedControl(
                                    id="segmented-gender-overview",
                                    value="all",
                                    data=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Female", "value": "female"},
                                        {"label": "Male", "value": "male"},
                                    ],
                                    radius="xl",
                                    size="sm",
                                ),
                            ]
                        ),

                        html.Div(className="toolbar-divider"),

                        # Year range slider (with mono range label above)
                        html.Div(
                            [
                                dmc.Group(
                                    [
                                        html.Div("TIME RANGE", className="toolbar-eyebrow"),
                                        html.Div(f"{1901}–{lastyearincluded}", id="overview-range-label",
                                                 className="toolbar-range"),
                                    ],
                                    justify="space-between",
                                ),
                                dmc.RangeSlider(
                                    id="slider-timerange-overview",
                                    value=[1901, lastyearincluded],
                                    min=1901,
                                    max=lastyearincluded,
                                    minRange=1,
                                    label=None,
                                    className="spectrum-slider",
                                    mt=6,
                                ),
                            ],
                            style={"flex": "1 1 260px", "minWidth": "240px"},
                        ),
                    ],
                    align="flex-end",
                    gap="lg",
                    wrap="wrap",
                ),
                className="filter-toolbar",
            ),
            # This is where the content goes; it is customized by the callback responding to the controls. See further down in the code in the callbacks section.
            # The various children elements will get stacked, i.e. arranged vertically.
            dmc.Stack(
                children=[],
                id="overview-content",
            )
        ]
    )





# Dynamic Content for the Overview Tab based of the selected filter values
# ##################################################################################################

@app.callback(
    Output("overview-range-label", "children"),
    Input("slider-timerange-overview", "value"),
)
def update_overview_range_label(value):
    return f"{value[0]}–{value[1]}"


@app.callback(
    Output("overview-content", "children"),
    [   Input("chip-medicine", "checked"),
        Input("chip-physics", "checked"),
        Input("chip-chemistry", "checked"),
        Input("chip-economics", "checked"),
        Input("chip-literature", "checked"),
        Input("chip-peace", "checked"),
        Input("segmented-gender-overview", "value"),
        Input("slider-timerange-overview", "value"),
        Input("theme-store", "data")],
        # prevent_initial_call=True
)
def update_overview_content(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, gender_value, timerange, theme):
    dark = (theme == "dark")

    selected_categories = pdg.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    # print("Selected Categories:", selected_categories)
    # Map the segmented control (all/female/male) back to the two gender flags.
    chip_female = gender_value in ("all", "female")
    chip_male = gender_value in ("all", "male")
    selected_gender = pdg.define_gender_states(chip_female, chip_male)

    df_filtered_laureates = pdg.standard_filter(data=df_laureates, categories=selected_categories, gender=selected_gender, timerange=timerange, callsign="overview callback filter laureates")
    df_filtered_prizes = pdg.standard_filter(data=df_prizes, categories=selected_categories, gender=selected_gender, timerange=timerange, callsign="overview callback filter prizes")

    number_of_laureates, number_of_prizes, laureate_oldest_name, laureate_oldest_age, laureate_youngest_name, laureate_youngest_age = pdg.generate_overview_stats(df_filtered_laureates, df_filtered_prizes)
    # print("Number of Laureates:", number_of_laureates)
    # print("Number of Prizes:", number_of_prizes)
    # print("Oldest Laureate:", laureate_oldest_name, laureate_oldest_age)
    # print("Youngest Laureate:", laureate_youngest_name, laureate_youngest_age)

    # Return the Content
    return [
        dcc.Loading(
            children=[
            # First row: Info tiles
            dmc.SimpleGrid(
                cols={"base": 1, "xs": 2, "md": 4},
                spacing={"base": "sm", "sm": "sm"},
                verticalSpacing={"base": "sm", "sm": "sm"},
                children=[
                    html.Div(
                        [
                            # Top part (Title)
                            html.Div(
                                [
                                    html.H4("Number of Prizes"),
                                    html.P("excluding declined prizes"),
                                ],
                                className="widget-title",
                            ),
                            # Bottom part (Content)
                            html.Div(
                                html.H1(number_of_prizes)
                            ),
                        ],
                        className="widget-container",
                    ),

                    html.Div(
                        [
                            # Top part (Title)
                            html.Div(
                                [
                                    html.H4("Number of Laureates"),
                                    html.P("including organisations"),
                                ],
                                className="widget-title",
                            ),
                            # Bottom part (Content)
                            html.Div(
                                html.H1(number_of_laureates)
                            ),
                        ],
                        className="widget-container",
                    ),

                    html.Div(
                        [
                            # Top part (Title)
                            html.Div(
                                [
                                    html.H4("Youngest Laureate"),
                                    html.P("excluding organisations"),
                                ],
                                className="widget-title",
                            ),
                            # Bottom part (Content)
                            html.Div(
                                [
                                    html.H1(laureate_youngest_age),
                                    html.H4(laureate_youngest_name)
                                ]
                            ),
                        ],
                        className="widget-container",
                    ),

                    html.Div(
                        [
                            # Top part (Title)
                            html.Div(
                                [
                                    html.H4("Oldest Laureate"),
                                    html.P("excluding organisations"),
                                ],
                                className="widget-title",
                            ),
                            # Bottom part (Content)
                            html.Div(
                                [
                                    html.H1(laureate_oldest_age),
                                    html.H4(laureate_oldest_name)
                                ]
                            ),
                        ],
                        className="widget-container",
                    ),
                ]
            ),
            ]
        ),

        # Bento row: Sunburst (wide, span 7) + Spektrum-Verteilung (narrow, span 5)
        dmc.Grid(
            columns=12,
            gutter="sm",
            children=[
                dmc.GridCol(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Discipline - Gender - Country"),
                                    html.P("Click on the segments to filter the data"),
                                ],
                                className="widget-title",
                            ),
                            html.Div(
                                dcc.Graph(
                                    id="fig_sunburst_overview",
                                    figure=pdg.generate_sunburst(data=df_filtered_laureates, dark=dark),
                                    style={'width': '100%', 'height': '100%'},
                                    config={"displayModeBar": False},
                                ),
                                className="widget-content",
                            ),
                        ],
                        className="widget-container",
                        style={"height": "100%"},
                    ),
                    span={"base": 12, "md": 7},
                ),
                dmc.GridCol(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Spectrum Distribution"),
                                    html.P("Prizes per category in the current selection"),
                                ],
                                className="widget-title",
                            ),
                            spektrum_verteilung(df_filtered_prizes, dark=dark),
                        ],
                        className="widget-container",
                        style={"height": "100%"},
                    ),
                    span={"base": 12, "md": 5},
                ),
            ],
        ),

        # Donut row: three equal-sized donuts
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="sm",
            verticalSpacing="sm",
            children=[
                html.Div(
                    [
                        # Top part (Title)
                        html.Div(
                            [
                                html.H4("Gender"),
                                html.P("According to the official Nobel API data"),
                            ],
                            className="widget-title",
                        ),
                        # Bottom part (Content)
                        html.Div(
                            dcc.Graph(id="fig_donut_gender_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="gender", dark=dark), style={'width': '100%', 'height':'100%'}, config={"displayModeBar": False}),
                            className="widget-content-ar1",
                        ),
                    ],
                    className="widget-container",
                ),

                html.Div(
                    [
                        # Top part (Title)
                        html.Div(
                            [
                                html.H4("Ethnicity"),
                                html.P("See the tab Data for more details." ),
                            ],
                            className="widget-title",
                        ),
                        # Bottom part (Content)
                        html.Div(
                            dcc.Graph(id="fig_donut_ethnicity_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="ethnicity", dark=dark), style={'width': '100%', 'height':'100%'}, config={"displayModeBar": False}),
                            className="widget-content-ar1",
                        ),
                    ],
                    className="widget-container",
                ),

                html.Div(
                    [
                        # Top part (Title)
                        html.Div(
                            [
                                html.H4("Religion"),
                                html.P("See the tab Data for more details."),
                            ],
                            className="widget-title",
                        ),
                        # Bottom part (Content)
                        html.Div(
                            dcc.Graph(id="fig_donut_religion_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="religion", dark=dark), style={'width': '100%', 'height':'100%'}, config={"displayModeBar": False}),
                            className="widget-content-ar1",
                        ),
                    ],
                    className="widget-container",
                ),

            ]
        ),

        # Third row: Stats
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 1},
            spacing="sm",
            verticalSpacing="sm",
            children=[
                html.Div(
                    [
                        # Top part (Title)
                        html.Div(
                            [
                                html.H4("Prize Statistics"),
                                html.P("More Interesting Facts"),
                            ],
                            className="widget-title",
                        ),
                        # Bottom part (Content)
                        html.Div([ag_df_prizestats],
                            className="widget-content"
                        ),
                        html.Div(
                            [
                                html.P("The columns 1/1, 1/2 and 1/3 refer between how many laureates the prize was shared: one single recipient, two recipients, or three."),
                            ],
                            className="widget-footer",
                        ),
                    ],
                    className="widget-container",
                ),

            ]
        ),

    ]




##################################################################################################
# Section Current
##################################################################################################

# Full category label shown on the 2025 cards (canonical key -> display label / raw data value).
_CURRENT_CAT_LABEL = {
    "Medicine": "Physiology or Medicine",
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Economics": "Economic Sciences",
    "Literature": "Literature",
    "Peace": "Peace",
}
_CURRENT_CAT_RAW = {
    "Medicine": "Medicine",
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Economics": "Economic Sciences",
    "Literature": "Literature",
    "Peace": "Peace",
}


def get_current_prizes_structured():
    """Per-category laureates + motivation for the most recent award year, in canonical order."""
    df_cur = df_prizes[df_prizes["Prize0_AwardYear"] == lastyearincluded]
    result = {}
    total = 0
    for cat in th.CATEGORY_ORDER:
        sub = df_cur[df_cur["Prize0_Category"] == _CURRENT_CAT_RAW[cat]].sort_index()
        names = list(dict.fromkeys(sub["AwardeeDisplayName"].tolist()))
        motivations = [m for m in sub["Prize0_Motivation"].tolist() if isinstance(m, str) and m.strip()]
        motivation = max(motivations, key=len) if motivations else ""
        result[cat] = {"names": names, "motivation": motivation}
        total += len(names)
    return result, total


def prize_card(category, names, motivation):
    """One category card for the 2025 Prizes page: accent bar + icon + names + motivation."""
    return html.Div(
        [
            html.Div(
                [msym(th.CATEGORY_ICONS[category]), html.Span(_CURRENT_CAT_LABEL[category].upper())],
                className="prize-cat-label",
            ),
            html.Div(" · ".join(names) if names else "—", className="prize-names"),
            html.Div(motivation, className="prize-motivation") if motivation else None,
        ],
        className=f"prize-card cat-{category.lower()}",
    )


def render_current_content():
    prizes, total = get_current_prizes_structured()
    cards = [prize_card(cat, prizes[cat]["names"], prizes[cat]["motivation"]) for cat in th.CATEGORY_ORDER]
    return dmc.Stack(
        children=[
            # Header row: title + laureate count pill
            dmc.Group(
                [
                    html.Div(
                        [
                            html.Div(f"Laureates {lastyearincluded}", className="page-title"),
                            html.Div("The Nobel Prize laureates of the most recent award year",
                                     className="page-subtitle"),
                        ]
                    ),
                    html.Div(f"{total} laureates", className="laureate-count-pill"),
                ],
                justify="space-between",
                align="center",
            ),
            # Per-category card grid (three columns)
            dmc.SimpleGrid(
                cols={"base": 1, "sm": 2, "md": 3},
                spacing="md",
                verticalSpacing="md",
                children=cards,
            ),
            generate_loader_spinner("fig_sunburst_last"),
            generate_loader_spinner("fig_map_movement"),
        ],
        gap="md",
    )


##################################################################################################
# Section Geography
##################################################################################################

def render_geography_content():
    return dmc.Stack(
        children=[
            generate_loader_spinner("fig_choroplethglobe_prizespercountry"),
            generate_loader_spinner("fig_map_cities"),
            generate_loader_spinner("fig_bubbles_population"),
            generate_loader_spinner("fig_bar_prizespercountry"),
            generate_loader_spinner("fig_bar_prizespercountry_rs"),
        ],
        gap="lg"
    )

##################################################################################################
# Section Demography
##################################################################################################

def render_demography_content():
    return dmc.Stack(
        children=[
            generate_loader_spinner("fig_surface_prizesforwomen"),
            generate_loader_spinner("fig_surface_prizesformenwomen"),
            generate_loader_spinner("fig_donut_gender"),
            generate_loader_spinner("fig_donut_ethnicity"),
            generate_loader_spinner("fig_donut_religion"),
        ],
        gap="sm"
    )

##################################################################################################
# Section Time
##################################################################################################

def render_time_content():
    return dmc.Stack(
        children=[
            generate_loader_spinner("fig_histogram_timegap"),
            generate_loader_spinner("fig_scatter_timegap_trend"),
            generate_loader_spinner("fig_scatterbox_age"),
            generate_loader_spinner("fig_heatmap_age"),
        ],
        gap="sm"
    )

##################################################################################################
# Section Misc
##################################################################################################

def render_misc_content():
    return dmc.Stack(
        children=[

            # PLOT: Nobel Fields
            generate_png_in_layout(   
                header = "Nobel Fields",
                subheader = "Cube sizes represent the number of Nobel prizes awarded to that field.",
                filepath = "/assets/images/fig_cubes_fields.png",
                style = {'width':'1000px'},
                footer = [
                    dcc.Markdown("**Interesting Findings**: Researchers with a momentum strategy should focus their research on particle physics or immunology, while contrarians should choose ethnology or chaos theory.")
                    ],
            ),
            generate_loader_spinner("fig_mostcommon_firstnames"),
            generate_loader_spinner("fig_line_prizemoney")
        ],
        gap="sm"
    )



##################################################################################################
# Section Migration
##################################################################################################

def render_migration_content():
    return dmc.Stack(
        children=[
            generate_loader_spinner("fig_parcat_migration_dwp"),
            generate_loader_spinner("fig_parcat_migration_bpd"),
            generate_loader_spinner('fig_globe_movement')
        ],
        gap="sm"
    )

##################################################################################################
# Section Nominations
##################################################################################################

def render_nominations_content():
    return dmc.Stack(

        children=[
            dmc.Alert(
                "Network graph generation can be very slow, depending on selected filters and the algorithm chosen. When trying a new algorithm, always start with a small number of selected nominations (<100).",
                title="Warning",
                withCloseButton=True,
            ),
            generate_loader_spinner("fig_network_nominations"),
            generate_loader_spinner("fig_map_nominations"),
        ],
        gap="sm"
    )

##################################################################################################
# Section List Generator
##################################################################################################

def render_listgenerator_content():

    return dmc.Stack(
        children=[

            # 1st inner stack for the filter options
            dmc.Stack(
                children = [

                    # 1st row
                    html.Div("Prize Categories"),
                    html.Div(
                        dmc.Group(
                            [
                                dmc.Chip("Medicine", checked=True, color="medicine", id="chip-medicine"),
                                dmc.Chip("Physics", checked=True, color="physics", id="chip-physics"),
                                dmc.Chip("Chemistry", checked=True, color="chemistry", id="chip-chemistry"),
                                dmc.Chip("Economics", checked=True, color="economics", id="chip-economics"),
                                dmc.Chip("Literature", checked=True, color="literature", id="chip-literature"),
                                dmc.Chip("Peace", checked=True, color="peace", id="chip-peace")
                            ]
                        )
                    ),

                    dmc.Divider(
                        variant="dotted", 
                        color="gray", 
                        size="xs",
                        style={"width": "100%"}
                    ),


                    # 2nd row                    
                    dmc.Grid(
                        style={"width": "100%"},
                        children=[
                            dmc.GridCol(
                                dmc.Stack(
                                    children=[
                                        html.Div("Gender"),
                                        html.Div(
                                            dmc.Group(
                                                [
                                                    dmc.Chip("female", variant="outline", checked=True, color="medicine", id="chip-female"),
                                                    dmc.Chip("male", variant="outline", checked=True, color="peace", id="chip-male"),
                                                ]
                                            )
                                        )
                                    ]
                                ),
                                span={'base': 24, 'md': 6}
                            ),

                            dmc.GridCol(       
                                dmc.Stack(
                                    children=[
                                        html.Div("Type"),
                                        html.Div(
                                            dmc.Group(
                                                [
                                                    dmc.Chip("Humans", variant="outline", checked=True, id="chip-humans"),
                                                    dmc.Chip("Organizations", variant="outline", checked=True, id="chip-organizations"),
                                                ]
                                            )
                                        )
                                    ]
                                ),
                                span={'base': 24, 'md': 8}
                            ),  

                            dmc.GridCol(  
                                dmc.Stack(
                                    children=[
                                        html.Div("Alive"),
                                        html.Div(
                                            dmc.Group(
                                                [
                                                    dmc.Chip("Alive", variant="outline", checked=True, id="chip-alive"),
                                                    dmc.Chip("Dead", variant="outline", checked=True, id="chip-dead"),
                                                ]
                                            )
                                        )
                                    ]
                                ),
                                span={'base': 24, 'md': 6}
                            ),  # End Alive/Dead Chip Stack

                            dmc.GridCol(  
                                dmc.Stack(
                                    children=[
                                        html.Div("Number of Prizes"),
                                        html.Div(
                                            [
                                                dmc.Slider(
                                                    id="slider-numberofprizes",
                                                    value=1,
                                                    min=1,
                                                    max=3,
                                                    step=1,                       
                                                    marks=[
                                                        {"value": 1},
                                                        {"value": 2},
                                                        {"value": 3}
                                                    ],
                                                    color="peace",
                                                    # style={"width": "50"}, 
                                                    mt=10
                                                ),
                                            ],
                                        )  # End number of prizes Slider
                                    ]
                                ),  # End Number of Prizes Slider Stack
                                span={'base': 24, 'md': 4}
                            ),  # End Number of Prizes Slider Stack


                        ],
                        columns=24
                    ),  # End of Line 2 Grid

                    dmc.Divider(
                        variant="dotted", 
                        color="gray", 
                        size="xs",
                        style={"width": "100%"}
                    ),


                    #3rd row
                    dmc.SimpleGrid(
                        cols={"base": 1, "sm": 2, "lg": 2},
                        spacing="xl",
                        style={"width": "100%"},
                        children=[

                            dmc.Stack(
                                children=[

                                    html.Div("Countries of Birth"),
                                    html.Div(
                                        [
                                            dmc.MultiSelect(
                                                # label="Select your favorite libraries",
                                                placeholder="Select countries",
                                                id="ms-countriesofbirth",
                                                #value=["pd", "torch"],
                                                data=pdg.countries_to_list(column="BirthCountryNow"),
                                                clearable=True,
                                                searchable=True,
                                                #w=400,
                                                mb=10,
                                            ),
                                            dmc.Text(id="ms-countriesofbirth-text"),
                                        ]
                                    ),



                                ]
                            ),

                            dmc.Stack(
                                children=[
                                    html.Div("Countries of Affiliation at the Time of the Award"),
                                    html.Div(
                                        [
                                            dmc.MultiSelect(
                                                #label="Select your favorite libraries",
                                                placeholder="Select countries",
                                                id="ms-countriesofaffiliation",
                                                #value=["pd", "torch"],
                                                data=pdg.countries_to_list(column="Prize0_Affiliation0_CountryNow"),
                                                clearable=True,
                                                searchable=True,
                                                #w=400,
                                                mb=10,
                                            ),
                                            dmc.Text(id="ms-countriesofaffiliation-text"),
                                        ]
                                    )


                                ]
                            )



                        ]
                    ),

                    dmc.Divider(
                        variant="dotted", 
                        color="gray", 
                        size="xs",
                        style={"width": "100%"}
                    ),


                    # 4th row
                    dmc.SimpleGrid(
                        cols={"base": 1, "sm": 2, "lg": 2},
                        spacing="xl",
                        style={"width": "100%"},
                        children=[
                            dmc.Stack(
                                children=[
                                    html.Div("Year of Birth"),
                                    html.Div(
                                        [
                                            dmc.RangeSlider(
                                                id="slider-timerange-birth",
                                                value=[1817, lastyearincluded],
                                                min=1817,
                                                max=lastyearincluded-25,
                                                minRange=1,
                                                marks=[
                                                    #{"value": 1817, "label": "1817"},
                                                    {"value": 1825, "label": "1825"},
                                                    {"value": 1850, "label": "1850"},
                                                    {"value": 1875, "label": "1875"},
                                                    {"value": 1900, "label": "1900"},
                                                    {"value": 1925, "label": "1925"},
                                                    {"value": 1950, "label": "1950"},
                                                    {"value": 1975, "label": "1975"},
                                                    {"value": 2000, "label": "2000"},
                                                    #{"value": int(lastyearincluded), "label": lastyearincluded}
                                                ],
                                                mb=35,
                                                color="peace"
                                            ),

                                        ],
                                    )
                                ],
                            ),

                            dmc.Stack(
                                children=[
                                    html.Div("Year of Award"),
                                    html.Div(
                                        [
                                            dmc.RangeSlider(
                                                id="slider-timerange-award",
                                                value=[1901, lastyearincluded],
                                                min=1901,
                                                max=lastyearincluded,
                                                minRange=1,
                                                marks=[
                                                    {"value": 1901, "label": "1901"},
                                                    {"value": 1925, "label": "1925"},
                                                    {"value": 1950, "label": "1950"},
                                                    {"value": 1975, "label": "1975"},
                                                    {"value": 2000, "label": "2000"},
                                                    {"value": int(lastyearincluded), "label": lastyearincluded}
                                                ],
                                                mb=35,
                                                color="peace"
                                            ),
                                        ],
                                    )  # End Time Range Slider
                                ],
                            )  # End Stack for Time Range Slider
                        ]
                    ),


                    dmc.Divider(
                        variant="dotted", 
                        color="gray", 
                        size="xs",
                        style={"width": "100%"}
                    ),



                    # 5th row
                    dmc.Group(
                        children=[
                            dmc.Stack(
                                children=[
                                    html.Div("Free Text Search on Prize Motivation"),
                                    html.Div(
                                        [
                                            dmc.TagsInput(
                                                # label="Select frameworks",
                                                placeholder="Enter free text (e.g. 'quantum dot')",
                                                id="motivation-input",
                                                # value=["ng", "vue"],
                                                # w=400,
                                                mb=10,
                                            )
                                        ]
                                    )
                                ]
                            ),  # End Stack for Prize Motivation Search

                            dmc.Stack(
                                children=[
                                    html.Div("Search Mode"),
                                    html.Div(
                                        [
                                            dmc.SegmentedControl(
                                                id="search-mode",
                                                value="any",
                                                data=[
                                                    {"value": "any", "label": "any term"},
                                                    {"value": "all", "label": "all terms"},
                                                ],
                                                # mb=10,
                                                pb=0,
                                            ),
                                        ]
                                    )
                                ]
                            )  # End Stack for Output Options


                        ]
                    ),



                    dmc.Divider(
                        variant="dotted", 
                        color="gray", 
                        size="xs",
                        style={"width": "100%"}
                    ),

                    # 6th row
                    dmc.Stack(
                        children=[
                            html.Div("List Output Options"),
                            html.Div(
                                [
                                    dmc.SegmentedControl(
                                        id="output-options",
                                        value="compact",
                                        data=[
                                            {"value": "names", "label": "Names Only"},
                                            {"value": "compact", "label": "Compact"},
                                            {"value": "extended", "label": "Extended"},
                                            {"value": "full", "label": "Everything"},
                                        ],
                                        pb=-1,
                                    ),
                                ]
                            )
                        ]
                    )  # End Stack for Output Options

                ],
                gap="sm",
                className="selection-area"
            ),  # End Filtering Area


            # 2nd inner stack for the results
            # This is where the content goes; it is customized by the callback responding to the controls. 
            dmc.Stack(
                children=[],
                id="listgenerator-content",
            )

        # End outer stack
        ]
    )


# Dynamic Content for the List Generator Section based of the selected filter values
# ##################################################################################################

@app.callback(
    Output("listgenerator-content", "children"), 
    [   Input("chip-medicine", "checked"),
        Input("chip-physics", "checked"),
        Input("chip-chemistry", "checked"),
        Input("chip-economics", "checked"),
        Input("chip-literature", "checked"),
        Input("chip-peace", "checked"),
        Input("chip-female", "checked"),
        Input("chip-male", "checked"),
        Input("chip-humans", "checked"),
        Input("chip-organizations", "checked"),
        Input("chip-alive", "checked"),
        Input("chip-dead", "checked"),
        Input("slider-numberofprizes", "value"),
        Input("ms-countriesofbirth", "value"),
        Input("ms-countriesofaffiliation", "value"),
        Input("slider-timerange-birth", "value"),
        Input("slider-timerange-award", "value"),
        Input("motivation-input", "value"),
        Input("search-mode", "value"),
        Input("output-options", "value")
    ],
        # prevent_initial_call=True
)
def update_listgenerator_content(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, chip_female, chip_male, chip_humans, chip_organizations, chip_alive, chip_dead, slider_numberofprizes, ms_countriesofbirth, ms_countriesofaffiliation, timerange_birth, timerange_award, motivation_input, search_mode, output_options):

    selected_categories = pdg.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    selected_gender = pdg.define_gender_states(chip_female, chip_male)
    selected_type = pdg.define_type_states(chip_humans, chip_organizations)
    selected_alive = pdg.define_alive_states(chip_alive, chip_dead)

    df_filtered_laureates = pdg.extended_filter(categories=selected_categories, gender=selected_gender, type=selected_type, alive=selected_alive, numberofprizes=slider_numberofprizes, countries_of_birth=ms_countriesofbirth, countries_of_affiliation=ms_countriesofaffiliation,  timerange_birth=timerange_birth, timerange_award=timerange_award, motivation_input=motivation_input, search_mode=search_mode, output_options=output_options)

    ag_df_listresults = dag.AgGrid(
        id="lg-aggrid",
        rowData=df_filtered_laureates.to_dicts(),  
        columnDefs=[{"field": i} for i in df_filtered_laureates.columns],
        defaultColDef={
            "filter": True, 
            "sortable": True, 
            "resizable": True,
            "maxWidth": 700,     # Maximale Breite
            "minWidth": 100,      # Minimale Breite
            "flex": 1            # Spalten fÃƒÂ¼llen verfÃƒÂ¼gbaren Platz
        },
        dashGridOptions={
            "pagination": True,
            "autoSizeStrategy": {
                "type": "fitGridWidth",  # Spalten fÃƒÂ¼llen Grid-Breite
                "defaultMinWidth": 100,
                "defaultMaxWidth": 500
            }
        },
        csvExportParams={
        "fileName": "nobel_laureates_filtered.csv",
        "allColumns": True,
        "suppressQuotes": False
    },
        style={
        "height": "60vh",  # 60% der Viewport-HÃƒÂ¶he
        "width": "100%"
    }
    )

    # Return the Content
    return dmc.Stack(
        children=[
            # Top part (Title)
            html.Div([
                dmc.Group([
                    html.H3(f"List Results ({df_filtered_laureates.shape[0]})"),
                    dmc.Button("Export CSV", id="export-csv-btn", variant="outline", size="sm")
                ], justify="space-between"),
                html.P("Please also see the section 'Data and References' for more explanation."),
            ], className="widget-title"),

            # Bottom part (Content)
            html.Div([ag_df_listresults],
                className="widget-content"
            )
        ]
    )


# Callback to trigger export
@app.callback(
    Output("lg-aggrid", "exportDataAsCsv"),
    Input("export-csv-btn", "n_clicks"),
    prevent_initial_call=True
)
def export_csv(n_clicks):
    return True






##################################################################################################
# Section Data
##################################################################################################

# Ethnicity
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';', encoding="utf8")

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';', encoding="utf8")

# # Defining the grid of AGGrid: The full data table
ag_df_laureates = dag.AgGrid(
    id="nl-aggrid",
    rowData=df_laureates.to_dict("records"),
    columnDefs=[{"field": i} for i in df_laureates.columns],
    defaultColDef={"filter": True},
    dashGridOptions={"pagination": True}
)

# # Defining the grid of AGGrid: Ethnicity
ag_df_ethnicity = dag.AgGrid(
    id="ethnicity-aggrid",
    rowData=df_ethnicity.to_dict("records"),
    columnDefs=[{"field": i} for i in df_ethnicity.columns],
    defaultColDef={"filter": True},
    dashGridOptions={"pagination": True}
)

# # Defining the grid of AGGrid: Religion
ag_df_religion = dag.AgGrid(
    id="religion-aggrid",
    rowData=df_religion.to_dict("records"),
    columnDefs=[{"field": i} for i in df_religion.columns],
    defaultColDef={"filter": True},
    dashGridOptions={"pagination": True}
)

def render_data_content():
    return dmc.Stack(
        id="section-data",
        children=[
            html.H3("About", className="text-header"),
            html.Div(dcc.Markdown(["The Nobel Laureate Data Dashboard is a project of Wolfgang Huang. If you want to learn more about my other projects, please see my portfolio page www.virtuousvector.ai ([Link](https://www.virtuousvector.ai)), or email me at mail-at-virtuousvector.ai."]), className="text-copy"),
            dmc.Space(h="xl"),
            
            html.H3("Data", className="text-header"),

            html.H5("Nobel Laureate Base Data", className="text-subheader"),
            html.Div("The core of the data is provided by Nobel Prize Outreach via their API. You can view it below, or access it via the API yourself. By the way, you can also sort and filter the data by clicking on the column headers / the column burger menu.", className="text-copy"),
            dmc.Space(h="xl"),
            html.Div([ag_df_laureates]),

            dmc.Space(h="xl"),

            html.H5("Timegap Analysis", className="text-subheader"),
            html.Div("The data used for timegap analysis is published here:", className="text-copy"),
            html.Div(dcc.Markdown(["Li, Jichao; Yin, Yian; Fortunato, Santo; Wang Dashun, 2018, \"A dataset of publication records for Nobel laureates\", Harvard Dataverse, [Link](https://doi.org/10.7910/DVN/6NJ5RN)"]), className="text-copy"),

            dmc.Space(h="xl"),

            html.H5("Degree - Work - Prize Migration Analysis", className="text-subheader"),
            html.Div("The data used for migration analysis is published here:", className="text-copy"),
            html.Div(dcc.Markdown(["Schlagberger, E.M., Bornmann, L. & Bauer, J.: \"At what institutions did Nobel laureates do their prize-winning work? An analysis of biographical information on Nobel laureates from 1994 to 2014\". Scientometrics 109, 723Ã¢â‚¬â€œ767 (2016). [Link](https://doi.org/10.1007/s11192-016-2059-2)"]), className="text-copy"),

            dmc.Space(h="xl"),

            html.H5("Population & Life Expectancy", className="text-subheader"),
            html.Div("The data used for population numbers and life expectancy is published here:", className="text-copy"),
            html.Div(dcc.Markdown(["Gapminder.org Data Downloads [Link](https://www.gapminder.org/data/)"]), className="text-copy"),


            dmc.Space(h="xl"),

            html.H5("Ethnicity", className="text-subheader"),
            html.Div("The data used for ethnicity is self-compiled. Further details are provided alongside the plot. Feel free to contact me if you have constructive criticism.", className="text-copy"),
            dmc.Space(h="xl"),
            html.Div([ag_df_ethnicity]),

            dmc.Space(h="xl"),

            html.H5("Religion", className="text-subheader"),
            html.Div("The data used for religion is self-compiled. Further details are provided alongside the plot. Feel free to contact me if you have constructive criticism.", className="text-copy"),
            dmc.Space(h="xl"),
            html.Div([ag_df_religion]),

            dmc.Space(h="xl"),

            html.H5("Download the Data", className="text-subheader"),
            html.Div(dcc.Markdown(["You can download all the data from my Github repository \"nobeldashboard\". [Link](https://github.com/WolfgangHuang/nobeldashboard)"]), className="text-copy"),
            dmc.Space(h="xl"),

            html.H5("Data Version", className="text-subheader"),
            html.Div([
                html.Span("The data was last updated via the official Nobel API at: "),
                html.Span(id="last-update-timestamp", children=get_last_update_timestamp())
            ], className="text-copy"),
            dcc.Interval(
                id='timestamp-interval',
                interval=60*1000,  # Update every minute
                n_intervals=0
            ),
            dmc.Space(h="xl"),

            html.H5("Version History", className="text-subheader"),
            html.Div(dcc.Markdown(["Version 2.0 (December 2025): Final nominations network graph & map"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.9 (October 2025): Automated API calls, nominations network"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.8 (July 2025): List generator, new layout, section URLs"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.7 (February 2025): Unified hover labels style, new color scheme"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.6 (January 2025): New plot 'Most Common Firstnames', design updates"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.5 (January 2025): Finalized New Filters"]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.4 (January 2025): New Filters using pattern matching; rewrote plot configs as class instances."]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.3 (December 2024): Rewrote all plots as functions, removed pickle storage."]), className="text-copy"),
            html.Div(dcc.Markdown(["Version 1.2 (November 2024): Design Upgrades / Improved Layout."]), className="text-copy"),
            dmc.Space(h="xl"),
        ]
    )




##################################################################################################
# Navigation Setup
##################################################################################################

# ------------------------------------------------------------------------------------------------
# Redesign frame helpers (Nobel-Spektrum)
# ------------------------------------------------------------------------------------------------

def msym(name, cls=""):
    """Render a Material Symbols Rounded glyph by name."""
    return html.Span(name, className=("material-symbols-rounded " + cls).strip())


def spectrum_logo(size="md"):
    """The six-stripe spectrum mark (pure CSS bars, canonical category order)."""
    cls = "nbl-logo" + (" sm" if size == "sm" else "")
    return html.Div(
        [html.Div(className=f"bar bar-{c.lower()}") for c in th.CATEGORY_ORDER],
        className=cls,
    )


# Grouped navigation. Each item carries a Material Symbol icon and a category accent.
NAV_SECTIONS = [
    ("ANALYSE", [
        {"label": "Overview",          "value": "overview",       "url": "/overview",     "icon": "dashboard",      "accent": "Economics"},
        {"label": f"{lastyearincluded} Prizes", "value": "current", "url": "/current",     "icon": "emoji_events",   "accent": "Medicine"},
        {"label": "Geography",         "value": "geography",      "url": "/geography",    "icon": "public",         "accent": "Physics"},
        {"label": "Demography",        "value": "demography",     "url": "/demography",   "icon": "groups",         "accent": "Peace"},
        {"label": "Time Analysis",     "value": "time",           "url": "/time",         "icon": "calendar_month", "accent": "Literature"},
        {"label": "Migration",         "value": "migration",      "url": "/migration",    "icon": "swap_horiz",     "accent": "Chemistry"},
        {"label": "Nominations",       "value": "nominations",    "url": "/nominations",  "icon": "campaign",       "accent": "Economics"},
        {"label": "Miscellaneous",     "value": "misc",           "url": "/misc",         "icon": "auto_awesome",   "accent": "Medicine"},
    ]),
    ("WERKZEUGE", [
        {"label": "List Generator",    "value": "list_generator", "url": "/list",         "icon": "table_chart",    "accent": "Physics"},
        {"label": "Data & References", "value": "data",           "url": "/data",         "icon": "storage",        "accent": "Peace"},
    ]),
]

# Flat list kept for URL <-> section mapping helpers below.
nav_items = [item for _, items in NAV_SECTIONS for item in items]


def build_navbar(active_value="overview"):
    """Render the grouped sidebar with the active item highlighted."""
    children = []
    for section_label, items in NAV_SECTIONS:
        children.append(html.Div(section_label, className="nbl-nav-section-label"))
        for item in items:
            is_active = item["value"] == active_value
            accent = th.SPECTRUM_LIGHT[item["accent"]]
            children.append(
                dcc.Link(
                    html.Div(
                        [msym(item["icon"]),
                         html.Span(item["label"], className="nbl-nav-label")],
                        className="nbl-nav-item" + (" active" if is_active else ""),
                        style={"--nav-accent": accent} if is_active else {},
                    ),
                    href=item["url"],
                    style={"textDecoration": "none"},
                )
            )
    return children

# URL-Mapping Funktionen
def get_section_from_url(pathname):
    """Konvertiert URL-Pfad zu Section-Name"""
    url_to_section = {item["url"]: item["value"] for item in nav_items}
    return url_to_section.get(pathname, "overview")

def get_url_from_section(section):
    """Konvertiert Section-Name zu URL-Pfad"""
    section_to_url = {item["value"]: item["url"] for item in nav_items}
    return section_to_url.get(section, "/overview")

def render_section_content(section):
    """Rendert Content basierend auf Section-Name"""
    if section == "overview":
        return render_overview_content()
    elif section == "current":
        return render_current_content()
    elif section == "geography":
        return render_geography_content()
    elif section == "demography":
        return render_demography_content()
    elif section == "time":
        return render_time_content()
    elif section == "migration":
        return render_migration_content()
    elif section == "nominations":
        return render_nominations_content()
    elif section == "misc":
        return render_misc_content()
    elif section == "list_generator":
        return render_listgenerator_content()
    elif section == "data":
        return render_data_content()
    else:
        return render_overview_content()

##################################################################################################
# AppShell Layout
##################################################################################################

_subtitle = f"1901–{lastyearincluded} · {numberofprizes} prizes · {df_laureates.shape[0]} laureates"

layout = dmc.AppShell(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        dcc.Store(id="sidebar-collapsed", data=False),
        dmc.AppShellHeader(
            dmc.Group(
                [
                    dmc.Group(
                        [
                            dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                            dmc.Box(
                                html.Button(msym("menu"), id="nav-collapse-btn",
                                            className="nbl-icon-btn", n_clicks=0,
                                            style={"background": "transparent"}),
                                visibleFrom="sm",
                            ),
                            spectrum_logo("md"),
                            html.Div(
                                [
                                    html.Div("Nobel Laureate Data Dashboard", className="nbl-wordmark"),
                                    html.Div(_subtitle, className="nbl-subtitle"),
                                ]
                            ),
                        ],
                        gap="sm",
                        align="center",
                    ),
                    dmc.Group(
                        [
                            dmc.TextInput(
                                id="laureate-search",
                                placeholder="Search laureates…",
                                leftSection=msym("search"),
                                radius="xl",
                                className="nbl-search",
                                visibleFrom="md",
                                w=220,
                            ),
                            html.Button(msym("dark_mode"), id="theme-toggle",
                                        className="nbl-icon-btn", n_clicks=0),
                            html.Div("V2.0", className="nbl-badge"),
                        ],
                        gap="sm",
                        align="center",
                    ),
                ],
                h="100%",
                px="lg",
                justify="space-between",
                align="center",
            ),
        ),
        dmc.AppShellNavbar(
            id="navbar",
            children=[
                html.Div(build_navbar("overview"), id="navbar-nav", className="nbl-navbar"),
            ],
            p="sm",
        ),
        dmc.AppShellMain(
            dmc.Container(
                html.Div(id="main-content", children=render_overview_content()),
                fluid=True,
                p="md"
            )
        ),
    ],
    header={"height": 72},
    navbar={
        "width": 212,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(
    layout,
    id="mantine-provider",
    forceColorScheme="light",
    theme=th.build_mantine_theme(),
)

##################################################################################################
# Callbacks
##################################################################################################

# ---- Theme (light/dark) toggle, persisted to localStorage --------------------------------------
@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks, current):
    return "dark" if (current or "light") == "light" else "light"


@app.callback(
    Output("mantine-provider", "forceColorScheme"),
    Output("theme-toggle", "children"),
    Input("theme-store", "data"),
)
def apply_color_scheme(scheme):
    scheme = scheme or "light"
    icon = "light_mode" if scheme == "dark" else "dark_mode"
    return scheme, msym(icon)


# Re-theme every standard plot on a theme switch — works on the existing figures, so it
# also covers plots that no MATCH filter callback ever reaches (e.g. categories=False).
@app.callback(
    Output({"type": "plot", "index": ALL}, "figure", allow_duplicate=True),
    Input("theme-store", "data"),
    State({"type": "plot", "index": ALL}, "figure"),
    prevent_initial_call=True,
)
def retheme_all_plots(theme, figures):
    dark = (theme == "dark")
    return [
        th.retheme_dict(f, dark) if (f and isinstance(f, dict) and "data" in f) else dash.no_update
        for f in figures
    ]


# ---- Sidebar: mobile burger + desktop collapse -------------------------------------------------
@app.callback(
    Output("appshell", "navbar"),
    Output("navbar-nav", "className"),
    Input("burger", "opened"),
    Input("nav-collapse-btn", "n_clicks"),
    State("appshell", "navbar"),
)
def manage_navbar(burger_opened, collapse_clicks, navbar):
    collapsed_desktop = bool(collapse_clicks and collapse_clicks % 2 == 1)
    navbar["width"] = 72 if collapsed_desktop else 212
    navbar["collapsed"] = {"mobile": not burger_opened}
    nav_cls = "nbl-navbar collapsed" if collapsed_desktop else "nbl-navbar"
    return navbar, nav_cls


# ---- Keep the active nav item in sync with the URL ---------------------------------------------
@app.callback(
    Output("navbar-nav", "children"),
    Input("url", "pathname"),
)
def highlight_active_nav(pathname):
    first_segment = "/" + (pathname or "/overview").strip("/").split("/")[0]
    section = get_section_from_url(first_segment)
    return build_navbar(section)

# URL-basierter Navigation Callback (erweitert)
@app.callback(
    [Output("main-content", "children"),
     Output("url", "pathname")],
    [Input({"type": "nav-item", "index": ALL}, "n_clicks"),
     Input("url", "pathname")],
    prevent_initial_call=True
)
def update_main_content_and_url(nav_clicks, pathname):
    ctx = callback_context
    
    # Check if URL changed (direct navigation)
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'url.pathname':
        
        # Handle plot-specific URLs (e.g., /geography/fig_choroplethglobe_prizespercountry)
        path_parts = pathname.strip('/').split('/')
        if len(path_parts) >= 2:
            section = path_parts[0]
            # Convert URL section to internal section name
            section = get_section_from_url('/' + section)
            content = render_section_content(section)
            return content, dash.no_update
        
        # Handle section URLs
        else:
            section = get_section_from_url(pathname)
            content = render_section_content(section)
            return content, dash.no_update
    
    # Check if navigation was clicked
    elif ctx.triggered and 'nav-item' in ctx.triggered[0]['prop_id']:
        # Find which nav item was clicked
        for i, clicks in enumerate(nav_clicks):
            if clicks:
                section = nav_items[i]["value"]
                url = nav_items[i]["url"]
                content = render_section_content(section)
                return content, url
    
    # Default: show overview
    content = render_section_content("overview")
    return content, "/overview"

# Timestamp update callback
@app.callback(
    Output('last-update-timestamp', 'children'),
    Input('timestamp-interval', 'n_intervals')
)
def update_timestamp_display(n_intervals):
    """Update the last update timestamp display."""
    return get_last_update_timestamp()

##################################################################################################
# Running the app
##################################################################################################

if __name__ == '__main__':
    port = int(os.environ.get('PORT', APP_CONFIG['port']))
    debug_mode = os.environ.get('DEBUG', str(APP_CONFIG['debug'])).lower() == 'true'
    host = os.environ.get('HOST', APP_CONFIG['host'])
    app.run(host=host, port=port, debug=debug_mode)