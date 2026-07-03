"""
Filter drawer renderers (standard + nominations) — view code extracted from config.py.
"""

from dash import dcc, html
import dash_mantine_components as dmc

import config as cf




# FILTER DEFINITIONS


def render_filter_standard(plot_config):
    """
    Generate the filter drawer component for standard plots.
    
    Creates a Mantine Drawer containing filter controls for categories,
    gender, time range, and optional custom filters. The filter configuration
    is driven by the plot_config.show_filters dictionary.
    
    Args:
        plot_config (PlotConfig): Configuration object specifying:
            - show_filters: Which filter controls to display
            - chips_notchecked: Category chips unchecked by default
            - chips_disabled: Category chips that cannot be toggled
            - timerange: Default [start, end] years for slider
            - timerange_field: Date field for filtering
            
    Returns:
        dmc.Drawer: Filter panel component with all configured controls.
        
    Filter Controls:
        - Category chips: Medicine, Physics, Chemistry, Economics, Literature, Peace
        - Gender chips: Female, Male
        - Time range: Slider with year selection
        - Time field: Dropdown to select which date to filter
        - Custom filters: Plot-specific additional controls
        - Submit mode: Toggle for immediate vs. button-triggered updates
    """
    from plotdatagenerator import lastyearincluded
    
    marks_award = [
        {"value": 1901, "label": "1901"},
        {"value": 1925, "label": "1925"},
        {"value": 1950, "label": "1950"},
        {"value": 1975, "label": "1975"},
        {"value": 2000, "label": "2000"},
        {"value": int(lastyearincluded), "label": lastyearincluded}
    ]

    marks_life = [
        {"value": 1817, "label": "1817"},
        {"value": 1900, "label": "1900"},
        {"value": int(lastyearincluded), "label": lastyearincluded}
    ]


    # Build children list and filter out None values
    filter_children = [
        # Category filter
        dmc.Stack(
            children=[
                html.Div("Prize Categories"),
                html.Div(
                    dmc.Group([
                        dmc.Chip("Medicine", size="xs", variant="outline", checked=False if "Medicine" in plot_config.chips_notchecked else True, disabled=True if "Medicine" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-medicine", "index": plot_config.plot_id}),
                        dmc.Chip("Physics", size="xs", variant="outline", checked=False if "Physics" in plot_config.chips_notchecked else True, disabled=True if "Physics" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-physics", "index": plot_config.plot_id}),
                        dmc.Chip("Chemistry", size="xs", variant="outline", checked=False if "Chemistry" in plot_config.chips_notchecked else True, disabled=True if "Chemistry" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-chemistry", "index": plot_config.plot_id}),
                        dmc.Chip("Economics", size="xs", variant="outline", checked=False if "Economics" in plot_config.chips_notchecked else True, disabled=True if "Economics" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-economics", "index": plot_config.plot_id}),
                        dmc.Chip("Literature", size="xs", variant="outline", checked=False if "Literature" in plot_config.chips_notchecked else True, disabled=True if "Literature" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-literature", "index": plot_config.plot_id}),
                        dmc.Chip("Peace", size="xs", variant="outline", checked=False if "Peace" in plot_config.chips_notchecked else True, disabled=True if "Peace" in plot_config.chips_disabled else False, color="economics", id={"type": "chip-peace", "index": plot_config.plot_id}),
                    ])
                ),
            ],
            style={
                "marginRight": "30px",
                "display": "block" if plot_config.show_filters.get("categories", False) else "none"
            }
        ) if plot_config.show_filters.get("categories", False) else None,
        
        # Gender Filter - ALWAYS render (for callback compatibility), but hide if not needed
        dmc.Stack(
            children=[
                html.Div("Gender"),
                html.Div(
                    dmc.Group([
                        dmc.Chip("female", size="xs", variant="outline", checked=True, color="economics", id={"type": "chip-female", "index": plot_config.plot_id}),
                        dmc.Chip("male", size="xs", variant="outline", checked=True, color="economics", id={"type": "chip-male", "index": plot_config.plot_id}),
                    ])
                ),
            ],
            style={
                "display": "block" if plot_config.show_filters.get("gender", False) else "none"
            }
        ),

        # TimeRange Filter
        dmc.Stack(
            children=[
                html.Div("Time Range"),
                html.Div(
                    dmc.Group([
                        dcc.Dropdown(
                            id={"type": "dropdown-timerange", "index": plot_config.plot_id},
                            options=[
                                {'label': 'Year of Birth', 'value': 'birth'},
                                {'label': 'Year of Award', 'value': 'award'},
                                {'label': 'Year of Death', 'value': 'death'}
                            ],
                            value='award',
                            clearable=False,
                            style={"width": "200px"},
                        ),
                        dmc.RangeSlider(
                            id={"type":"slider-timerange", "index": plot_config.plot_id},
                            value=[1901, lastyearincluded],
                            min=1800,
                            max=lastyearincluded,
                            minRange=1,
                            marks=marks_life,
                            style={"width": "400px"},
                            color="peace"
                        ),
                    ], mb=35)
                )
            ],
            style={"margin-top": "0px", "width":"100%"}
        ),

        # Custom Filter (if provided)
        plot_config.show_filters.get("custom-filter"),

        dmc.Space(h="lg"),

        dmc.Switch(
            id={"type":"switch-update", "index": plot_config.plot_id},
            size="sm",
            radius="xl",
            label="Update plot only on Submit",
            checked=False
        )
    ]
    
    # Remove None values from the list
    filter_children = [child for child in filter_children if child is not None]

    return dmc.Drawer(
        title="Filter",
        position="right",
        id={"type": "filter-modal", "index": plot_config.plot_id},
        size="300px",
        style={"display": "block"},
        children=[
            dmc.Stack(
                filter_children,  # Use filtered list here
                gap="sm",
                className="selection-area",
            ),
            dmc.Group([
                dmc.Button("Submit", id={"type": "submit-button", "index": plot_config.plot_id}, color="economics"),
                dmc.Button("Close", color="c_red", variant="outline", id={"type": "close-button", "index": plot_config.plot_id}),
            ], justify="flex-end"),
        ],
    )                                            



def render_filter_nominations(plot_config):
    """
    Generate the filter drawer component for nomination plots.
    
    Creates a specialized Mantine Drawer for filtering nomination network
    graphs and maps. Includes additional controls for nominator/nominee
    filtering and network layout algorithms.
    
    Args:
        plot_config (PlotConfig): Configuration object with plot_category='nominations'.
            
    Returns:
        dmc.Drawer: Filter panel with nomination-specific controls.
        
    Filter Controls:
        - Prize category chips
        - Nominator filters: Gender, country, name search, laureate-only
        - Nominee filters: Gender, country, name search, laureate-only
        - Time range: Year of nomination slider
        - Algorithm selection: Network layout algorithm dropdown (graphviz, etc.)
        - SFDP parameters: K-value, repulsive force, overlap handling
        - Submit mode: Toggle for immediate vs. button-triggered updates
        
    Note:
        Nomination data has a 50-year secrecy period, so max year is
        lastyearincluded - 49.
    """

    from plotdatagenerator import lastyearincluded, countries_to_list, df_edges

    marks_award = [
        {"value": 1901, "label": "1901"},
        {"value": 1925, "label": "1925"},
        {"value": 1950, "label": "1950"},
        {"value": 1975, "label": "1975"},
        {"value": 2000, "label": "2000"},
        {"value": int(lastyearincluded), "label": lastyearincluded}
    ]

    marks_life = [
        {"value": 1817, "label": "1800"},
        {"value": 1900, "label": "1900"},
        {"value": int(lastyearincluded), "label": lastyearincluded}
    ]

    marks_nomination = [
        {"value": 1901, "label": "1901"},
        {"value": int(lastyearincluded)-49, "label": lastyearincluded-49}
    ]


    # Build children list
    filter_children = [
        # 1st row - Selected count
        dmc.Alert(
            #title="Selected",
            children=[
                dmc.Text(
                    id={"type": "nom-nomination-count", "index": plot_config.plot_id},
                    size="md",
                    #fw=700,
                )
            ],
            color="economics",
            variant="light",
            style={"margin-bottom": "10px"}
        ),

        # Prize Categories
        html.Div("Prize Categories"),
        html.Div(
            dmc.Group([
                dmc.Chip("Medicine", checked=True, color="medicine", id={"type": "nom-chip-medicine", "index": plot_config.plot_id}),
                dmc.Chip("Physics", checked=True, color="physics", id={"type": "nom-chip-physics", "index": plot_config.plot_id}),
                dmc.Chip("Chemistry", checked=True, color="chemistry", id={"type": "nom-chip-chemistry", "index": plot_config.plot_id}),
                dmc.Chip("Economics", checked=True, color="economics", id={"type": "nom-chip-economics", "index": plot_config.plot_id}),
                dmc.Chip("Literature", checked=True, color="literature", id={"type": "nom-chip-literature", "index": plot_config.plot_id}),
                dmc.Chip("Peace", checked=True, color="peace", id={"type": "nom-chip-peace", "index": plot_config.plot_id})
            ])
        ),

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Gender filters                   
        dmc.Grid(
            style={"width": "100%"},
            children=[
                dmc.GridCol(
                    dmc.Stack(
                        children=[
                            html.Div("Nominator Gender"),
                            html.Div(
                                dmc.Group([
                                    dmc.Chip("female", size="xs", variant="outline", checked=True, color="economics", id={"type": "nom-chip-nominator-female", "index": plot_config.plot_id}),
                                    dmc.Chip("male", size="xs", variant="outline", checked=True, color="economics", id={"type": "nom-chip-nominator-male", "index": plot_config.plot_id}),
                                ])
                            ),
                        ],
                        style={
                            "display": "block" if plot_config.show_filters.get("gender", False) else "none"
                        }
                    ),
                    span={'base': 24, 'md': 8}
                ),
                dmc.GridCol(       
                    dmc.Stack(
                        children=[
                            html.Div("Nominee Gender"),
                            html.Div(
                                dmc.Group([
                                    dmc.Chip("female", size="xs", variant="outline", checked=True, color="economics", id={"type": "nom-chip-nominee-female", "index": plot_config.plot_id}),
                                    dmc.Chip("male", size="xs", variant="outline", checked=True, color="economics", id={"type": "nom-chip-nominee-male", "index": plot_config.plot_id}),
                                ])
                            ),
                        ],
                        style={
                            "display": "block" if plot_config.show_filters.get("gender", False) else "none"
                        }
                    ),
                    span={'base': 24, 'md': 8}
                ), 
            ]
        ) if plot_config.show_filters.get("gender", False) else None,

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Country filters
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 1, "lg": 1},
            spacing="xl",
            style={"width": "100%"},
            children=[
                dmc.Stack(
                    children=[
                        html.Div("Nominator Countries"),
                        html.Div([
                            dmc.MultiSelect(
                                placeholder="Select countries",
                                id={"type": "nom-nominator-country", "index": plot_config.plot_id},
                                data=countries_to_list(data=df_edges, column="nominator_country"),
                                clearable=True,
                                searchable=True,
                                mb=10,
                            ),
                        ])
                    ]
                ),
                dmc.Stack(
                    children=[
                        html.Div("Nominee Country"),
                        html.Div([
                            dmc.MultiSelect(
                                placeholder="Select countries",
                                id={"type": "nom-nominee-country", "index": plot_config.plot_id},
                                data=countries_to_list(data=df_edges, column="nominee_country"),
                                clearable=True,
                                searchable=True,
                                mb=10,
                            ),
                        ])
                    ]
                )
            ]
        ),

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Time Range
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 1, "lg": 1},
            spacing="xl",
            style={"width": "100%"},
            children=[
                dmc.Stack(
                    children=[
                        html.Div("Time Range"),
                        html.Div(
                            dmc.Group([
                                dcc.Dropdown(
                                    id={"type": "nom-dropdown-timerange", "index": plot_config.plot_id},
                                    options=[
                                        {'label': 'Year of Nomination', 'value': 'nomination'},
                                    ],
                                    value='nomination',
                                    clearable=False,
                                    style={"width": "100%"},
                                ),
                                dmc.RangeSlider(
                                    id={"type": "nom-slider-timerange", "index": plot_config.plot_id},
                                    value=[1901, 1905],
                                    min=1901,
                                    max=lastyearincluded-49,
                                    minRange=1,
                                    marks=marks_nomination,
                                    style={"width": "100%"},
                                    color="peace"
                                ),
                            ], mb=35)
                        )
                    ],
                    style={"margin-top": "0px", "width":"100%"}
                )
            ]
        ),

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Nominator Name search
        dmc.Group(
            children=[
                dmc.Stack(
                    children=[
                        html.Div("Nominator Name"),
                        html.Div([
                            dmc.TagsInput(
                                placeholder="Enter free text (e.g. 'Heisenberg')",
                                id={"type": "nom-nominator-name-input", "index": plot_config.plot_id},
                                mb=10,
                            )
                        ])
                    ]
                ),
                dmc.Stack(
                    children=[
                        html.Div("Search Mode"),
                        html.Div([
                            dmc.SegmentedControl(
                                id={"type": "nom-nominator-searchmode", "index": plot_config.plot_id},
                                value="any",
                                data=[
                                    {"value": "any", "label": "any term"},
                                    {"value": "all", "label": "all terms"},
                                ],
                                pb=0,
                            ),
                        ])
                    ]
                ),
                dmc.Group(
                    children=[
                        html.Div("Laureate Only"),
                        dmc.Checkbox(
                            id={"type": "nom-nominator-laureate-checkbox", "index": plot_config.plot_id},
                            checked=False,
                        )
                    ]
                ),
            ]
        ),

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Nominee Name search
        dmc.Group(
            children=[
                dmc.Stack(
                    children=[
                        html.Div("Nominee Name"),
                        html.Div([
                            dmc.TagsInput(
                                placeholder="Enter free text (e.g. 'Heisenberg')",
                                id={"type": "nom-nominee-name-input", "index": plot_config.plot_id},
                                mb=10,
                            )
                        ])
                    ]
                ),
                dmc.Stack(
                    children=[
                        html.Div("Search Mode"),
                        html.Div([
                            dmc.SegmentedControl(
                                id={"type": "nom-nominee-searchmode", "index": plot_config.plot_id},
                                value="any",
                                data=[
                                    {"value": "any", "label": "any term"},
                                    {"value": "all", "label": "all terms"},
                                ],
                                pb=0,
                            ),
                        ])
                    ]
                ),
                dmc.Group(
                    children=[
                        html.Div("Laureate Only"),
                        dmc.Checkbox(
                            id={"type": "nom-nominee-laureate-checkbox", "index": plot_config.plot_id},
                            checked=False,
                        )
                    ]
                ),
            ]
        ),

        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Layout selection (dash-cytoscape in-browser layouts).
        # NOTE: id stays "nom-algorithm" so the existing wiring keeps working; the
        # value is now a Cytoscape layout name, not a graphviz/networkx algorithm.
        dmc.Stack(
            children=[
                html.Div("Layout:"),
                dcc.Dropdown(
                    id={"type": "nom-algorithm", "index": plot_config.plot_id},
                    options=[
                        {'label': 'Cola (live physics)', 'value': 'cola'},
                        {'label': 'fCoSE (force, run once)', 'value': 'fcose'},
                        {'label': 'CoSE (force, run once)', 'value': 'cose'},
                        {'label': 'Concentric (by degree)', 'value': 'concentric'},
                        {'label': 'Circle', 'value': 'circle'},
                        {'label': 'Breadthfirst (tree)', 'value': 'breadthfirst'},
                        {'label': 'Grid', 'value': 'grid'},
                    ],
                    value='cola',
                    clearable=False,
                    style={"width": "100%"}
                ),
                html.Div("Tip: with 'Cola', grab a node and drag — neighbours follow.",
                         style={"fontSize": "11px", "color": cf.c_grey}),
            ],
            gap="sm",
            align="flex-start",
            style={
                "display": "block" if plot_config.show_filters.get("algorithm", False) else "none",
                "width": "100%"
                }
        ),


        # SFDP parameters (conditionally shown)
        dmc.Stack(
            id={"type": "nom-sfdp-params-container", "index": plot_config.plot_id},
            children=[
                dmc.Stack(
                    children=[
                        html.Div("K Value (edge length):"),
                        dmc.Slider(
                            id={"type": "nom-sfdp-k-value", "index": plot_config.plot_id},
                            min=0.1, max=3.0, step=0.1, value=0.3,
                            marks=[
                                {"value": 0.3, "label": "0.3"},
                                {"value": 1.0, "label": "1.0"},
                                {"value": 2.0, "label": "2.0"},
                                {"value": 3.0, "label": "3.0"}
                            ],
                            color="peace",
                            style={"width": "100%"}
                        )
                    ],
                    gap="xs", mb=30
                ),
                
                dmc.Stack(
                    children=[
                        html.Div("Repulsive Force:"),
                        dmc.Slider(
                            id={"type": "nom-sfdp-rf-value", "index": plot_config.plot_id},
                            min=0.1, max=3.0, step=0.1, value=1.0,
                            marks=[
                                {"value": 0.3, "label": "0.3"},
                                {"value": 1.0, "label": "1.0"},
                                {"value": 2.0, "label": "2.0"},
                                {"value": 3.0, "label": "3.0"}
                            ],
                            color="peace",
                            style={"width": "100%"}
                        )
                    ],
                    gap="xs", mb=30
                ),
                dmc.Stack(
                    children=[
                        html.Div("Overlap Handling:"),
                        dmc.SegmentedControl(
                            id={"type": "nom-sfdp-overlap", "index": plot_config.plot_id},
                            value="scale",
                            data=[
                                {"value": "prism", "label": "prism"},
                                {"value": "scale", "label": "scale"},
                                {"value": "false", "label": "none"},
                            ],
                        ),
                    ],
                    gap="xs"
                ),
            ],
            gap="md",
            align="flex-start",
            style={"display": "none", "margin-top": "10px"}

        ) if plot_config.show_filters.get("algorithm", False) else None,


        # divider
        dmc.Divider(variant="dotted", color="gray", size="xs", style={"width": "100%"}),

        # Submit controls
        dmc.Stack(
            children=[
                dmc.Switch(
                    id={"type": "nom-switch-update", "index": plot_config.plot_id},
                    size="sm",
                    radius="xl",
                    label="Update plot only on Submit",
                    checked=True
                ),
            ]
        ),
    ]
    
    # Remove None values from the list
    filter_children = [child for child in filter_children if child is not None]

    return dmc.Drawer(
        title="Filter",
        position="right",
        id={"type": "nom-filter-modal", "index": plot_config.plot_id},
        size="300px",
        style={"display": "block"},
        children=[
            dmc.Stack(
                filter_children,
                gap="sm",
                className="selection-area",
            ),
            dmc.Group([
                dmc.Button("Submit", id={"type": "nom-submit-button", "index": plot_config.plot_id}, color="economics"),
                dmc.Button("Close", color="c_red", variant="outline", id={"type": "nom-close-button", "index": plot_config.plot_id}),
            ], justify="flex-end"),
        ],
    )


