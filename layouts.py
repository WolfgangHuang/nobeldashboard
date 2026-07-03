"""
Widget layout builders (loader spinner, standard/nominations plot shells, the
Cytoscape network widget). Extracted from app.py so config/plot_config no longer
need a circular `from app import ...`.
"""

from dash import dcc, html
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_cytoscape as cyto

import config as cf
import plotdatagenerator as pdg
import theme as th


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


NETWORK_DEFAULT_INFO = "Click on any person to highlight their nominations"


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
