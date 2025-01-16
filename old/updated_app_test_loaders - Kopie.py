##################################################################################################
# Library Imports
##################################################################################################

import pandas as pd
import dash_ag_grid as dag
import os
import dash
from dash import dcc, html, Dash, _dash_renderer, State, callback
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, MATCH
import dash_mantine_components as dmc
import pickle
import precompute_plots as pcp 



##################################################################################################
# General Settings
##################################################################################################

brand_color_plot_background='#FEFEFA'

c_brown = '#47382a'
c_brown_verylight = '#F0EBE6'
c_teal = '#186f77'
c_lightblue = '#93bbdc'
c_red = '#91121d'
c_orange = '#ff7703'
c_yellow = '#ffe74c'
c_darkmagenta = '#8e2984'
c_magenta = '#cf437d'
c_pink = '#ff99c8'

c_red_verylight = '#f7c1c6'
c_red_superlight = '#fbe0e2'
c_red_verydark = '#3A070B'

c_teal_verylight = '#C2EEF3'
c_teal_superlight = '#e1f7f9'
c_teal_verydark = '#072124'

c_lightblue_verylight = '#e9f1f8'
c_lightblue_superlight = '#f4f8fc'

c_pie1 = '#67d6e0'
c_pie2 = '#c9ddee'
c_pie3 = '#ec6570'
c_pie4 = '#ffbc82'
c_pie5 = '#fff3a6'
c_pie6 = '#da81d1'
c_pie7 = '#e7a2bf'
c_pie8 = '#ffcce4'
c_pie0 = '#b59b82'

colorscale_palette = [c_teal, c_lightblue, c_red, c_orange, c_yellow, c_darkmagenta, c_magenta, c_pink]
colorscale_palette_light = [c_pie1, c_pie2, c_pie3, c_pie4, c_pie5, c_pie6, c_pie7, c_pie8]

colorscale_red = [c_red_verylight, c_red_verydark]
colorscale_teal = [c_teal_verylight, c_teal_verydark]
colorscale_teal_log = ["#E1F7F9", "#67D6E0", "#67D6E0", "#186F77", "#13595F", "#114D53"]
colorscale_teal_to_read = [c_teal_verylight, c_teal, c_red]



colorscale_hue_log = [c_teal, c_lightblue]

brand_color_main = c_brown
brand_color_alt = c_teal
brand_color_alt2 = c_red
brand_color_acc = c_darkmagenta
brand_color_plot_background='#F7F5F2'
brand_colorscale_main = colorscale_palette
c_physics = c_teal
c_medicine = c_red
c_chemistry = c_orange
c_economics = c_lightblue
c_peace = c_pink
c_literature = c_yellow


##################################################################################################
# Load Precomputed Plots
##################################################################################################

# Get the path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to this location
os.chdir(current_dir)


# Load tables

# Laureates data
df_laureates = pd.read_csv('df_laureates_enriched_redux_clean.csv', sep=';', encoding="UTF-8", index_col=0)

# Same as laureates, but the two-time-winners are listed twice
df_prizes = pd.read_csv('df_prizes_enriched_redux_clean.csv', sep=';', encoding="UTF-8", index_col=0)

# Ethnicity
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';', encoding="UTF-8")

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';', encoding="UTF-8")



# Nobel Prize Stats
df_prizestats = pd.read_csv("df_prizestats.csv", sep=';', encoding="UTF-8")

df=pcp.count_per_country()
max_prize_count = df['Count'].max()
lastyearincluded = 2024
numberofprizes = df_prizes.shape[0]

##################################################################################################
# Dashboard Main Setup
##################################################################################################

_dash_renderer._set_react_version("18.2.0")

app = Dash(
    external_stylesheets=[
        "assets/dmc_styles.css",  # custom CSS
        dmc.styles.ALL           # Mantine styles
    ],
    title="Nobel Laureate Data Dashboard",
    suppress_callback_exceptions=True
)


##################################################################################################
# AG Grid Definitions
##################################################################################################

# https://medium.com/plotly/getting-started-with-dash-ag-grid-v-31-f167ee19083b
# https://dash.plotly.com/dash-ag-grid

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
]

ag_df_prizestats = dag.AgGrid(
    id="prizestats-aggrid",
    rowData=df_prizestats.to_dict("records"),
    #columnDefs=[{"field": i} for i in df_prizestats.columns],
    columnDefs = columnDefs,
    columnSize="sizeToFit",
    dashGridOptions={
        "pagination": False,
    }
)


##################################################################################################
# Function to generate plots in the layout
##################################################################################################
  
def generate_plot_in_layout(
    cols={"base": 1, "sm": 1},
    header="Generic Plot Title",
    subheader="",
    datafrom="1901",
    datato=lastyearincluded,
    badges=[dmc.Badge("All Categories", variant="outline", color=brand_color_alt)],
    code="",
    code2="",
    plot_generator=None,
    plot_id="fig_type_content",
    figure=None,
    style={'width': '100%', 'height': '100%'},
    footer="",
    content_classname="widget-content",
    filter_modal_id="modal-simple",
    submit_button_id="modal-submit-button"
):
    
    print(f"Generator: Generating plot with ID: {plot_id} and plot_generator: {plot_generator}")

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
                            html.H3(header),
                            html.P(subheader) if subheader else None,
                        ]
                    ),

                    # Filter Modal
                    html.Div(
                        [
                            dmc.Button("Filter", id={"type": "filter-button", "index": plot_id}),
                            dmc.Modal(
                                title="Filter",
                                centered=True,
                                id={"type": "filter-modal", "index": plot_id},
                                size="70%",
                                style={"display": "block"},
                                children=[
                                    # Selection Area
                                    dmc.Stack(
                                        [
                                            # Category filter
                                            dmc.Stack(
                                                children=[
                                                    html.Div("Select Categories"),
                                                    html.Div(
                                                        dmc.Group(
                                                            [
                                                                dmc.Chip("Medicine", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-medicine", "index": plot_id}),
                                                                dmc.Chip("Physics", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-physics", "index": plot_id}),
                                                                dmc.Chip("Chemistry", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-chemistry", "index": plot_id}),
                                                                dmc.Chip("Economics", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-economics", "index": plot_id}),
                                                                dmc.Chip("Literature", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-literature", "index": plot_id}),
                                                                dmc.Chip("Peace", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-peace", "index": plot_id}),
                                                            ]
                                                        )
                                                    ),
                                                ],
                                                style={"marginRight": "30px"}
                                            ),
                                            # Gender Filter
                                            dmc.Stack(
                                                children=[
                                                    html.Div("Select Gender"),
                                                    html.Div(
                                                        dmc.Group(
                                                            [
                                                                dmc.Chip("female", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-female", "index": plot_id}),
                                                                dmc.Chip("male", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-male", "index": plot_id}),
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            )
                                        ],
                                        gap="sm",
                                        className="selection-area",
                                    ),
                                    # filter modal buttons
                                    dmc.Group(
                                        [
                                            dmc.Button("Submit", id={"type": "submit-button", "index": plot_id}),
                                            dmc.Button(
                                                "Close",
                                                color="red",
                                                variant="outline",
                                                id={"type": "close-button", "index": plot_id},
                                            ),
                                        ],
                                        justify="flex-end",
                                    ),
                                ],
                            ),
                        ]
                    ),

                    # Custom Filter Code
                    dmc.Stack(
                        children=[
                            html.Div(code)
                        ]
                    ),

                    # Plot
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                # id=plot_id,
                                id={"type": "plot", "index": plot_id},
                                figure=figure,
                                style=style,
                            )
                        ),
                        className=content_classname,
                    ),

                    # Footer
                    html.Div(
                        [
                            *footer,
                        ],
                        className="widget-footer",
                    ) if footer else None,

                    # Store the plot generator function directly
                    dcc.Store(id={"type": "plot-generator", "index": plot_id}, data=plot_generator)
                ],
                className="widget-container",
            ),
        ],
    )

# # Callback to toggle the modal (close it)
@app.callback(
    Output({"type": "filter-modal", "index": MATCH}, "opened"),
    Input({"type": "filter-button", "index": MATCH}, "n_clicks"),
    Input({"type": "close-button", "index": MATCH}, "n_clicks"),
    Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
    State({"type": "filter-modal", "index": MATCH}, "opened"),
    prevent_initial_call=True,
)
def toggle_modal(nc1, nc2, nc3, opened):
    return not opened

# ..............................................................................
# # Functional callback with logging for debugging purposes
# # Callback to toggle the modal (close it)
# @app.callback(
#     Output({"type": "filter-modal", "index": MATCH}, "opened"),
#     [
#         Input({"type": "filter-button", "index": MATCH}, "n_clicks"),
#         Input({"type": "close-button", "index": MATCH}, "n_clicks"),
#         Input({"type": "submit-button", "index": MATCH}, "n_clicks")
#     ],
#     State({"type": "filter-modal", "index": MATCH}, "opened"),
#     prevent_initial_call=True,
# )
# def toggle_modal(nc1, nc2, nc3, opened):
#     ctx = dash.callback_context

#     if not ctx.triggered:
#         return opened
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]

#     # Log the button that triggered the callback
#     print(f"Modal: Button clicked: {button_id}")
#     print(f"Modal: nc1: {nc1}, nc2: {nc2}, nc3: {nc3}, opened: {opened}")

#     # Extract the `index` from the triggering component's ID
#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
#     triggered_id_dict = eval(triggered_id)  # Convert string to dictionary

#     # Log the `plot_id`
#     plot_id = triggered_id_dict.get("index")
#     print(f"Modal: callback triggered by {triggered_id} with plot_id={plot_id}")

#     # Toggle the modal state
#     return not opened
# ..............................................................................


# ..............................................................................
# Simple callback for debugging purposes
# @app.callback(
#     Output({"type": "plot", "index": MATCH}, "figure"),
#     Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
# )
# def test_callback(n_clicks):
#     print(f"Plot Update Callback triggered with {n_clicks} clicks")
#     return {}
# ..............................................................................



# updates the plot based on the selected filter values
# the index refers to the specific plot and is defined when calling the function
# the MATCH keyword states that the id's of the input provided via the modal and the plot to be updated must match
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
        Input({"type": "submit-button", "index": MATCH}, "n_clicks")
    ],
    [
        State({"type": "plot", "index": MATCH}, "figure"),
        State({"type": "plot-generator", "index": MATCH}, "data")
    ]
)
def update_plot(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, chip_female, chip_male, n_clicks, current_figure, plot_generator_name):
    # Ensure the callback is only triggered when the submit button is clicked
    if n_clicks is None:
        raise PreventUpdate

    # Define the selected categories and gender filters
    print(f"Updater: callback triggered with {n_clicks} clicks")
    print("Updater: Category states:", chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    print("Updater: Gender states:", chip_female, chip_male)
    print("Updater: Plot Generator:", plot_generator_name)

    selected_categories = pcp.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    selected_genders = pcp.define_gender_states(chip_female, chip_male)
    
    #plot_generator= pcp.generate_map_movement

    print(selected_categories)
    print (selected_genders)
    print(plot_generator_name)

    # Gets the plot generator function based on the name stored in the State, which is set in the layout generator function
    if plot_generator_name:
        plot_generator = getattr(pcp, plot_generator_name, None)
        if not plot_generator:
            raise ValueError(f"Function {plot_generator_name} not found in module!")
    else:
        raise ValueError("No plot generator name provided.")

    updated_figure = plot_generator(categories=selected_categories, gender=selected_genders)

    return updated_figure



# function to include pngs in the layout - should not be used, as the dashboard shoudl only include plotly figures
def generate_png_in_layout(
    cols= {"base": 1, "sm": 1},
    header= "Generic Plot Title", 
    subheader="", 
    datafrom="1901", 
    datato=lastyearincluded, 
    badges=[dmc.Badge("All Categories", variant="outline", color= brand_color_alt)],
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
                            html.H3(header),
                            html.P(subheader) if subheader else None,
                            dmc.Group(
                                [
                                    dmc.Badge(f"{datafrom} - {datato}", variant="outline", color="blue"),
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
# Layout Definition
##################################################################################################

app.layout = dmc.MantineProvider(
    children=[
        dmc.Container(
            children=[
                dmc.Grid(
                    children=[
                        dmc.GridCol(
                            dmc.Group(
                                [
                                    html.Img(src="assets/logo-md.png", style={"width": "150px", "height": "50px"}),
                                    html.H1("Nobel Laureate Data Dashboard v1.24", className="text-left mt-5 mb-5"),
                                ]
                            ),
                        span=12)
                    ]
                ),

                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("Overview", value="tab_overview"),
                                dmc.TabsTab(f"{lastyearincluded} Prizes", value="tab_current"),
                                dmc.TabsTab("Geography", value="tab_geography"),
                                dmc.TabsTab("Demography", value="tab_demography"),
                                dmc.TabsTab("Time", value="tab_time"),
                                dmc.TabsTab("Migration", value="tab_migration"),
                                dmc.TabsTab("Misc", value="tab_misc"),
                                dmc.TabsTab("Data & References", value="tab_data"),
                            ]
                        ),
                        # Unique IDs for each tab panel
                        
                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-overview",  # Unique ID for loading spinner
                                children=html.Div(id="tab-content-overview")  # Unique ID for tab content
                            ),
                            value="tab_overview"
                        ),
                        
                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-current",  # Unique ID for loading spinner
                                children=html.Div(id="tab-content-current")  # Unique ID for tab content
                            ),
                            value="tab_current"
                        ),

                       dmc.TabsPanel(
                            html.Div(id="tab-content-geography"),
                            value="tab_geography"
                        ),

                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-demography",
                                children=html.Div(id="tab-content-demography")
                            ),
                            value="tab_demography"
                        ),

                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-time",
                                children=html.Div(id="tab-content-time")
                            ),
                            value="tab_time"
                        ),

                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-migration",
                                children=html.Div(id="tab-content-migration")
                            ),
                            value="tab_migration"
                        ),

                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-misc",
                                children=html.Div(id="tab-content-misc")
                            ),
                            value="tab_misc"
                        ),

                        dmc.TabsPanel(
                            dcc.Loading(
                                id="tab-content-loading-data",
                                children=html.Div(id="tab-content-data")
                            ),
                            value="tab_data"
                        ),
                    ],
                    value="tab_current",  # Default selected tab
                    id="tabs"
                ),
                dcc.Store(id="background-loading-trigger", data=False),  # Trigger for background loading
                dcc.Store(id="preloaded-figures", data={}, storage_type="session") # Store for preloaded figures, max 5-8 MB
            ],
            fluid=True,
            style={"margin": "0px", "backgroundColor": "#ffffff", "maxWidth": "1200px"}
        )
    ]
)



##################################################################################################
##################################################################################################
# Callbacks for Initial Data Loading for Active Tabs
##################################################################################################
##################################################################################################


##################################################################################################
# Tab Overview
##################################################################################################


@app.callback(
    [Output("tab-content-overview", "children"), Output("background-loading-trigger", "data")],
    Input("tabs", "value")
)
def render_tab_overview_content(active_tab):
    if active_tab == 'tab_overview':

        # Define the content for tab2024
        content = dmc.Paper(
            children=[

                dmc.Stack(
                    [
                        html.Div("Select Categories"),
                        html.Div(
                            dmc.Group(
                                [
                                    dmc.Chip("Medicine", checked=True, color=c_medicine, id="chip-medicine"),
                                    dmc.Chip("Physics", checked=True, color=c_physics, id="chip-physics"),
                                    dmc.Chip("Chemistry", checked=True, color=c_chemistry, id="chip-chemistry"),
                                    dmc.Chip("Economics", checked=True, color=c_economics, id="chip-economics"),
                                    dmc.Chip("Literature", checked=True, color=c_literature, id="chip-literature"),
                                    dmc.Chip("Peace", checked=True, color=c_peace, id="chip-peace")
                                ]
                            )

                        ),

                        html.Div("Select Time Range"),
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    id='overview-timerange-control',
                                    step=1,
                                    value=[1901, int(lastyearincluded)],  # Default range from 0 to max
                                    marks={i: str(i) for i in range(1901, int(lastyearincluded+5), 5)}, 
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="dmc-bar dmc-thumb",
                                )
                            ],
                            #justify="left",
                            style={"margin-top": "0px", "width":"100%"}
                        )
               

                    ],
                    gap="sm",
                    className="selection-area"
                ),
                  # This is where the content goes; it is customized by the callback responding to the controls. See further down in the code in the callbacks section.
                  # The various children elements will get stacked, i.e. arranged vertically.
                  dmc.Stack(
                    children=[],
                    id="overview-content",
                ),

            ],
            shadow="lg",
            radius="lg",
            p="lg", 
            className="mt-3",
            style={"backgroundColor": "#ffffff"}
        )

        return content, True  # Return the content and set the pre-loading-trigger-status to TRUE, i.e. preloading can now start.

    else:
        return html.Div("No content available"), False


# Preloading
# ##################################################################################################

# This callback preloads one figure (bubbles) to a dcc.store with is placed at teh end of the layout.
# The dcc.store has an initial status of FALSE. 
# Once the loading of the first tab finishes, the status is set to TRUE, which triggers the callback below.
# It seems that the data is onyl kept for "one click", and after that it is forgotten. 
# This should be influencable by setting storage_type="session", but it doesn't seem to work.
# Will continue to monitor.

@app.callback(
    Output("preloaded-figures", "data"),
    Input("background-loading-trigger", "data"),
    prevent_initial_call=True,  # Only trigger when the trigger changes
)
def preload_figures(trigger):
    if trigger:
        # Preload figures from the pickle file
        with open("pcp_plots.pkl", "rb") as f:
            pcp_plots = pickle.load(f)

        # Extract figures
        preloaded_figures = {
            "fig_bubbles_population" : pcp_plots['fig_bubbles_population'],
        }
        return preloaded_figures
    return {}



# Dynamic Content for the Overview Tab
# ##################################################################################################

@app.callback(
    Output("overview-content", "children"), 
    [   Input("chip-medicine", "checked"),
        Input("chip-physics", "checked"),
        Input("chip-chemistry", "checked"),
        Input("chip-economics", "checked"),
        Input("chip-literature", "checked"),
        Input("chip-peace", "checked"),
        Input("overview-timerange-control", "value")],
        # prevent_initial_call=True
)
def update_overview_content(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, timerange):

    with open('pcp_plots.pkl', 'rb') as f2:
        pcp_plots = pickle.load(f2)
    
    # Extract the figures from the loaded pickle data
    fig_donut_gender = pcp_plots['fig_donut_gender']
    fig_donut_ethnicity = pcp_plots['fig_donut_ethnicity']
    fig_donut_religion = pcp_plots['fig_donut_religion']
    fig_sunburst = pcp_plots['fig_sunburst']

    selected_categories = pcp.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)

    df_filtered_prizes = pcp.standard_filter(df_laureates, selected_categories)
    df_filtered_laureates = pcp.standard_filter(df_prizes, selected_categories)


    # Number of Prizes
    # -------------------
    number_of_prizes = df_filtered_prizes.shape[0]


    # Number of Laureates#
    # ---------------------------
    number_of_laureates = df_filtered_laureates.shape[0]


    # Calculate Youngest and Oldest laureate
    # ---------------------------------------
   
    if df_filtered_prizes.shape[0] == 0:
        laureate_oldest_name = "None"
        laureate_oldest_age = ""
        laureate_youngest_name = "None"
        laureate_youngest_age = ""
    else:
        df_oldestyoungest_laureate = df_filtered_prizes[["AwardeeDisplayName", "OrganisationName", "BirthDate", "Prize0_AwardYear"]].copy()

        df_oldestyoungest_laureate = df_oldestyoungest_laureate[pd.isna(df_oldestyoungest_laureate["OrganisationName"])]

        df_oldestyoungest_laureate["BirthDate"] = df_oldestyoungest_laureate["BirthDate"].str.replace(r"-00-00", "-01-01", regex=True)

        df_oldestyoungest_laureate["BirthDate"] = pd.to_datetime(df_oldestyoungest_laureate["BirthDate"])

        # Convert Prize0_AwardYear to YYYY-12-10 format
        df_oldestyoungest_laureate["AwardDate"] = pd.to_datetime(
            df_oldestyoungest_laureate["Prize0_AwardYear"].astype(str) + "-12-10"
        )

        # Calculate the difference
        df_oldestyoungest_laureate["AgeAtAward"] = (
            df_oldestyoungest_laureate["AwardDate"] - df_oldestyoungest_laureate["BirthDate"]
        )

        from dateutil.relativedelta import relativedelta

        # Function to calculate exact age in years
        def calculate_exact_years(row):
            if pd.isna(row["BirthDate"]) or pd.isna(row["AwardDate"]):
                return None  # Handle missing dates gracefully
            return relativedelta(row["AwardDate"], row["BirthDate"]).years

        # Apply the function to calculate age in years
        df_oldestyoungest_laureate["AgeAtAwardYears"] = df_oldestyoungest_laureate.apply(calculate_exact_years, axis=1)

        df_sorted = df_oldestyoungest_laureate.sort_values(by="AgeAtAward", ascending=False)

        laureate_oldest_name = df_sorted.iloc[0]["AwardeeDisplayName"]
        laureate_oldest_age = df_sorted.iloc[0]["AgeAtAwardYears"]

        #print(f"{laureate_oldest_name}: {laureate_oldest_age}")

        df_sorted = df_oldestyoungest_laureate.sort_values(by="AgeAtAward", ascending=True)

        laureate_youngest_name = df_sorted.iloc[0]["AwardeeDisplayName"]
        laureate_youngest_age = df_sorted.iloc[0]["AgeAtAwardYears"]

        #print(f"{laureate_youngest_name}: {laureate_youngest_age}")
        
    
    # Update figures

    fig_donut_gender = pcp.generate_donut(data=df_filtered_laureates, characteristic="gender")
    fig_donut_ethnicity = pcp.generate_donut(data=df_filtered_laureates, characteristic="ethnicity")
    fig_donut_religion = pcp.generate_donut(data=df_filtered_laureates, characteristic="religion")
    fig_sunburst = pcp.generate_sunburst(data=df_filtered_laureates)

    # Return the Content
    # ----------------------------------------------

    return [
        
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

        # Second row: graph widgets
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
                            dcc.Graph(id="fig_donut_gender", figure=fig_donut_gender, style={'width': '100%', 'height':'100%'}),
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
                            dcc.Graph(id="fig_donut_gender", figure=fig_donut_ethnicity, style={'width': '100%', 'height':'100%'}),
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
                            dcc.Graph(id="fig_donut_religion", figure=fig_donut_religion, style={'width': '100%', 'height':'100%'}),
                            className="widget-content-ar1",
                        ),
                    ],
                    className="widget-container",
                ),

            ]
        ),

        # Third row: Sunburst
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
                                html.H4("Discipline - Gender - Country"),
                                html.P("Click on the segments to filter the data"),
                            ],
                            className="widget-title",
                        ),
                        # Bottom part (Content)
                        html.Div(
                            dcc.Graph(id="fig_sunburst", figure=fig_sunburst, style={'width': '100%', 'height':'100%'}),
                            className="widget-content",
                        ),
                    ],
                    className="widget-container",
                ),

            ]
        ),

        # Fourth row: Stats
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
# Tab Current
##################################################################################################


@app.callback(
    Output('tab-content-current', 'children'),
    Input('tabs', 'value')
)
def render_tab_current_content(active_tab):
    if active_tab == 'tab_current':
        # Load precomputed plots only if they haven't been loaded yet
        with open('pcp_plots.pkl', 'rb') as f1:
            pcp_plots = pickle.load(f1)
        
        # Extract the figures from the loaded pickle data
        fig_sunburst_last = pcp_plots['fig_sunburst_last']
        fig_map_movement = pcp_plots['fig_map_movement']


        # Return the content for tab2024
        return dmc.Paper(
            children=[
                dmc.Stack(
                    children=[
                        dmc.SimpleGrid(
                            cols={"base": 1, "xs": 2, "md": 4},
                            spacing={"base": "sm", "sm": "sm"},
                            verticalSpacing={"base": "sm", "sm": "sm"},
                            children=
                                [

                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Physiology or Medicine")
                                            ],
                                            className="widget-title",
                                            #style={"borderBottom": f"1px solid {c_medicine}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "Victor Ambros",
                                                    ),
                                                    html.H4(
                                                        "Gary Ruvkun",
                                                    ),
                                                    html.P(
                                                        "for the discovery of microRNA and its role in post-transcriptional gene regulation",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),


                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Physics")
                                            ],
                                            className="widget-title",
                                            #style={"borderBottom": f"1px solid {c_physics}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "John J. Hopfield",
                                                    ),
                                                    html.H4(
                                                        "Geoffrey Hinton",
                                                    ),
                                                    html.P(
                                                        "for foundational discoveries and inventions that enable machine learning with artificial neural networks",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),


                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Chemistry")
                                            ],
                                            className="widget-title",
                                            # style={"borderBottom": f"1px solid {c_chemistry}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "David Baker"
                                                    ),
                                                    html.P(
                                                        "for computational protein design",
                                                    ),
                                                    html.H4(
                                                        "Demis Hassabis",
                                                    ),
                                                    html.H4(
                                                        "John N. Jumper",
                                                    ),
                                                    html.P(
                                                        "for protein structure prediction",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),

                            ]
                        ),


                        dmc.SimpleGrid(
                            cols={"base": 1, "xs": 2, "md": 4},
                            spacing={"base": "sm", "sm": "sm"},
                            verticalSpacing={"base": "sm", "sm": "sm"},
                            children=
                                [

                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Literature")
                                            ],
                                            className="widget-title",
                                            #style={"borderBottom": f"1px solid {c_medicine}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "Han Kang",
                                                    ),
                                                    html.P(
                                                        "for her intense poetic prose that confronts historical traumas and exposes the fragility of human life",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),


                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Peace")
                                            ],
                                            className="widget-title",
                                            #style={"borderBottom": f"1px solid {c_physics}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "Nihon Hidankyo",
                                                    ),
                                                    html.P(
                                                        "for its efforts to achieve a world free of nuclear weapons and for demonstrating through witness testimony that nuclear weapons must never be used again",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),


                                html.Div(
                                    [
                                        # Top part (Title)
                                        html.Div(
                                            [
                                                html.H4("Economic Sciences")
                                            ],
                                            className="widget-title",
                                            #style={"borderBottom": f"1px solid {c_chemistry}"}
                                        ),
            
                                        html.Div(
                                            children=[
                                                    html.H4(
                                                        "Daron Acemoglu",
                                                    ),
                                                    html.H4(
                                                        "Simon Johnson",
                                                    ),
                                                    html.H4(
                                                        "James A. Robinson",
                                                    ),
                                                    html.P(
                                                        "for studies of how institutions are formed and affect prosperity",
                                                    )
                                            ],
                                            className="widget-content"
                                        ),
                                    ],
                                    className="widget-container",
                                ),

                            ]
                        ),

                        
                        generate_plot_in_layout(   
                            header = "Discipline - Gender - Country",
                            subheader = "Click on the segments to filter the data.",
                            plot_id = "fig_sunburst_last",
                            figure = fig_sunburst_last,
                            plot_generator = "generate_sunburst",
                            footer = [
                                dcc.Markdown("**Interesting Findings**: The 2024 prizes were mostly given to male researchers from the US or UK.")
                                ],
                        ),

                        generate_plot_in_layout(   
                            header = "Life Paths (Birth - Work)",
                            subheader = "Some Laureates actually haven't moved and are represented as dots.",
                            plot_id = "fig_map_movement",
                            figure = fig_map_movement,
                            plot_generator = "generate_map_movement",
                            footer = [
                                dcc.Markdown("**Interesting Findings**: Common patterns: European researchers move to the US, the Americans switch coasts at most, and the Asians stay where they are. No South Americans or Africans.")
                                ],
                        ),

                    ],
                    gap="sm"
                )
            ],
            shadow="lg",
            radius="lg",
            p="lg", 
            className="mt-3",
            style={"backgroundColor": "#ffffff"}
        )

    else:
        return html.Div("No content available")



##################################################################################################
# Tab Geography
##################################################################################################

@app.callback(
    Output("tab-content-geography", "children"),  # Output for the tab content
    Input("tabs", "value"),  # Active tab
    State("preloaded-figures", "data"),  # Preloaded figures
)
def render_geography_tab(active_tab, preloaded_figures):
    if active_tab == 'tab_geography':

        required_figures = [
            "fig_choroplethglobe_prizespercountry",
            "fig_map_cities",
            "fig_bubbles_population",
            "fig_bar_prizespercountry",
            "fig_bar_prizespercountry_rs",
        ]

        # Check if all required figures are in the preloaded figures
        missing_figures = [fig for fig in required_figures if fig not in preloaded_figures]

        if missing_figures:
            # Dynamically load missing figures
            with open("pcp_plots.pkl", "rb") as f:
                pcp_plots = pickle.load(f)

            # Add missing figures to the preloaded dictionary
            for fig in missing_figures:
                preloaded_figures[fig] = pcp_plots[fig]
        
        

        # Return the content for tab Nationality
        return dmc.Paper(
            children=[
                html.Div("This tab contains plots related to the nationality of Nobel Laureates. Actually, most of the time, it refers to the country of birth, which is unique and easy to identify, unlike the actual nationality. Note that quite some Laureates will actually not have the nationality of the country they were born in."),
                dmc.Space(h="xl"),

                dmc.Stack(
                    children=[
                        
                        # # PLOT: Rotatable Globe
                        # generate_plot_in_layout(   
                        #     header = "Nobel Prizes by Country of Birth",
                        #     subheader = "This plot shows the distribution of country of birth of the laureates; you may rotate the globe, and zoom in and out. The slider lets you select minimum and maximum number of laureates, e.g.\"*less than 5 laureates*\".",
                        #     plot_id = "fig_choroplethglobe_prizespercountry",
                        #     figure = fig_choroplethglobe_prizespercountry,
                        #     footer = [
                        #         dcc.Markdown("**Interesting Findings**: The US dominance is clearly visible; apart from Africa, there are surprisingly few white spots.")
                        #         ],
                        #     code= dmc.Grid(
                        #         children=[
                        #             dmc.GridCol(
                        #                 dcc.RangeSlider(
                        #                     id='fig_choroplethglobe_year_slider',
                        #                     min=1901,
                        #                     max=lastyearincluded,  # Dynamic maximum value based on data
                        #                     step=1,
                        #                     value=[0, lastyearincluded],  # Default range from 0 to max
                        #                     marks={i: str(i) for i in range(0, int(lastyearincluded) + 1, 50)}, 
                        #                     tooltip={"placement": "bottom", "always_visible": True},
                        #                     className="dmc-bar dmc-thumb",
                        #                 ),
                        #                 span=6
                        #             ),
                        #         ],
                        #         justify="left",
                        #         style={"margin-top": "20px"}
                        #     ),
                        # ),

       
                        # # PLOT: Map: Places of Birth and Death
                        # generate_plot_in_layout(   
                        #     header = "Places of Birth and Death",
                        #     subheader = f"This map shows the cities of birth and death of the laureates. Note that the points of the map are given as center points of the respective cities, not as the actual places of birth (e.g. hospitals). You can zoom in to quite some detail; the map data is provided via OpenStreetMap.",
                        #     plot_id = "fig_map_cities",
                        #     figure = fig_map_cities,
                        #     code=   dmc.Group(
                        #                 children=[
                        #                     html.P("Please select location:", style={"margin-top": "5px", "font-weight": "bold"}),
                        #                     dcc.Dropdown(
                        #                         id='fig_map_cities_city_dropdown',
                        #                         options=[
                        #                             {'label': 'City of Birth', 'value': 'birth'},
                        #                             {'label': 'City of Affiliation at Time of Award', 'value': 'affiliation'}
                        #                             {'label': 'City of Death', 'value': 'death'}
                        #                         ],
                        #                         value='birth',  # Default value
                        #                         clearable=False,
                        #                         style={"width": "200px"}
                        #                     ),
                        #                 ],
                        #             gap="md",  # Adjusts the space between the label and the dropdown
                        #             align="flex-start",  # Align items to the left
                        #             style={"margin-top": "20px"}
                        #         ),
                        # ),     


                        html.Div(
                            id="outer-container-bubbles-population",
                            children=[
                                dmc.Loader(
                                    id="loader-bubbles-population",
                                    color= c_physics,
                                    size="md",  # Available sizes: xs, sm, md, lg, xl
                                    variant="dots",  # Available variants: oval, dots, bars
                                ),
                                html.Div(
                                    id="inner-container-bubbles-population",
                                    style={"display": "none"},  # Initially hidden
                                ),
                            ],
                            #style={"position": "relative", "width": "100%", "height": "70vh"},
                        ),

                        html.Div(
                            id="outer-container-bar-prizespercountry",
                            children=[
                                dmc.Loader(
                                    id="loader-bar-prizespercountry",
                                    color= c_physics,
                                    size="md",  # Available sizes: xs, sm, md, lg, xl
                                    variant="bars",  # Available variants: oval, dots, bars
                                ),
                                html.Div(
                                    id="inner-container-bar-prizespercountry",
                                    style={"display": "none"},  # Initially hidden
                                ),
                            ],
                            #style={"position": "relative", "width": "100%", "height": "70vh"},
                        ),



                        html.Div(
                            id="outer-container-bar-prizespercountry-rs",
                            children=[
                                dmc.Loader(
                                    id="loader-bar-prizespercountry-rs",
                                    color= c_physics,
                                    size="md",  # Available sizes: xs, sm, md, lg, xl
                                    variant="oval",  # Available variants: oval, dots, bars
                                ),
                                html.Div(
                                    id="inner-container-bar-prizespercountry-rs",
                                    style={"display": "none"},  # Initially hidden
                                ),
                            ],
                            #style={"position": "relative", "width": "100%", "height": "70vh"},
                        ),


                    ],
                    gap="sm"
                )

            ]
        )

    else:
        return html.Div("Data is loading.")

# Loader Callbacks
#######################################################################################


@callback(
    Output("inner-container-bubbles-population", "children"),
    Output("loader-bubbles-population", "style"),  # Hide loader
    Output("inner-container-bubbles-population", "style"),  # Show graph
    Input("inner-container-bubbles-population", "id"),  # Trigger on app load
    State("preloaded-figures", "data"),  # Access preloaded figures
)
def display_fig_bubbles_population(_, preloaded_figures):
    # Check if the figure is available in the preloaded data
    if preloaded_figures and "fig_bubbles_population" in preloaded_figures:
        fig_bubbles_population = preloaded_figures["fig_bubbles_population"] 
    else:
        # Load the figure dynamically if not preloaded
        with open("pcp_plots.pkl", "rb") as f1:
            pcp_plots = pickle.load(f1)
        fig_bubbles_population = pcp_plots["fig_bubbles_population"]

    # Generate the graph layout
    fig = generate_plot_in_layout(
        header="Nobel Prizes by Country of Birth and Population",
        subheader=(
            "This plot shows the number of prizes by country of birth, but in relation to the population size "
            "of the country in the respective year. You can use the slider or play button to see the animation of "
            "the years."
        ),
        plot_id="fig_bubbles_population",
        figure=fig_bubbles_population,
        footer=[
            dcc.Markdown(
                "**Interesting Findings**: The visualization shows, for example, that countries like the USA only "
                "start to play an important role after World War II; also, in relation to their population number, "
                "they did not get unusually many Nobel Prizes."
            )
        ],
        style={'width': 'auto', 'height': '70vh'},
    )

    # Hide the loader and show the graph
    return fig, {"display": "none"}, {"display": "block"}



@callback(
    Output("inner-container-bar-prizespercountry", "children"),
    Output("loader-bar-prizespercountry", "style"),  # Hide loader
    Output("inner-container-bar-prizespercountry", "style"),  # Show graph
    Input("inner-container-bar-prizespercountry", "id"),  # Trigger on app load
    State("preloaded-figures", "data"),  # Access preloaded figures

)
def display_fig_bar_prizespercountry(_, preloaded_figures):
    
    # Check if the figure is available in the preloaded data
    if preloaded_figures and "fig_bar_prizespercountry" in preloaded_figures:
        fig_bar_prizespercountry = preloaded_figures["fig_bar_prizespercountry"]     
    else:
        # Load the figure dynamically if not preloaded
        with open("pcp_plots.pkl", "rb") as f1:
            pcp_plots = pickle.load(f1)
        fig_bar_prizespercountry = pcp_plots["fig_bar_prizespercountry"]
    
    # Generate the graph layout
    fig = generate_plot_in_layout(   
                            header = "Nobel Prizes by Country of Birth per Year",
                            subheader = "This plot shows the number of prizes per country of birth of laureates by year. You may deselect and reselect countries from the legend to customize your plot.",
                            plot_id = "fig_bar_prizespercountry",
                            figure = fig_bar_prizespercountry,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: The many pink lines on top in the right part of the plot emphasize our earlier finding that the number of prizes given to the USA has increased tremendously only after World War II.")
                                ],
                        ),

    # Hide the loader and show the graph
    return fig, {"display": "none"}, {"display": "block"}


@callback(
    Output("inner-container-bar-prizespercountry-rs", "children"),
    Output("loader-bar-prizespercountry-rs", "style"),  # Hide loader
    Output("inner-container-bar-prizespercountry-rs", "style"),  # Show graph
    Input("inner-container-bar-prizespercountry-rs", "id"),  # Trigger on app load
    State("preloaded-figures", "data"),  # Access preloaded figures
)
def display_fig_bar_prizespercountry_rs(_, preloaded_figures):
    
    # Check if the figure is available in the preloaded data
    if preloaded_figures and "fig_bar_prizespercountry_rs" in preloaded_figures:
        fig_bar_prizespercountry_rs = preloaded_figures["fig_bar_prizespercountry_rs"] 
    else:
        # Load the figure dynamically if not preloaded
        with open("pcp_plots.pkl", "rb") as f1:
            pcp_plots = pickle.load(f1)
        fig_bar_prizespercountry_rs = pcp_plots["fig_bar_prizespercountry_rs"]

    # Generate the graph layout
    fig = generate_plot_in_layout(   
                            header = "Nobel Prizes by Country of Birth per Year",
                            subheader = "This plot shows the number of prizes per country of birth of laureates by year. You may deselect and reselect countries from the legend to customize your plot.",
                            plot_id = "fig_bar_prizespercountry_rs",
                            figure = fig_bar_prizespercountry_rs,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: The many pink lines on top in the right part of the plot emphasize our earlier finding that the number of prizes given to the USA has increased tremendously only after World War II.")
                                ],
                        ),

    # Hide the loader and show the graph
    return fig, {"display": "none"}, {"display": "block"}



##################################################################################################
# Tab Demography
##################################################################################################

@app.callback(
    Output('tab-content-demography', 'children'),
    Input('tabs', 'value')
)
def render_tab_demographics_content(active_tab):
    if active_tab == 'tab_demography':
        with open('pcp_plots.pkl', 'rb') as f4:
            pcp_plots = pickle.load(f4)
        
        # Extract the figures from the loaded pickle data
        fig_surface_prizesforwomen = pcp_plots['fig_surface_prizesforwomen']
        fig_surface_prizesformenwomen = pcp_plots['fig_surface_prizesformenwomen']
        fig_donut_gender = pcp_plots['fig_donut_gender']
        fig_donut_ethnicity = pcp_plots['fig_donut_ethnicity']
        fig_donut_religion = pcp_plots['fig_donut_religion']


        # Return the content for tab Demography
        return dmc.Paper(
            children=[
                # html.H5("Tab 4: Full Data", className="card-title"),
                html.Div(dcc.Markdown(["This tab contains plots related to gender, ethnicity and religion."])),
                dmc.Space(h="xl"),

                dmc.Stack(
                    children=[


                        # PLOT: 3D Surface: Nobel Prizes Awarded to Women
                        generate_plot_in_layout(   
                            header = "Nobel Prizes Awarded to Women",
                            subheader = "This plot shows the number of prizes for women in all of the disciplines per decade. It allows you to see when and in which disciplines the most prizes were awarded to women. Feel free to rotate the plot and zoom.",
                            plot_id = "fig_surface_prizesforwomen",
                            figure = fig_surface_prizesforwomen,
                            # footer = [
                            #     dcc.Markdown("**Interesting Findings**: See plot below.")
                            #     ],
                        ),

                        # PLOT: 3D Surface: Nobel Prizes Awarded to Men and Women
                        generate_plot_in_layout(   
                            header = "Nobel Prizes Awarded to Men and Women",
                            subheader = "This plot is identical to the above, but here, men and women are both shown as two surfaces.",
                            plot_id = "fig_surface_prizesformenwomen",
                            figure = fig_surface_prizesformenwomen,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: It is not much of a surprise that much many prizes were given to men than to women. Disciplines that perform particularly poorly are economics sciences and physics. The plot also clearly shows that the number of laureates (per year/decade) increases, i.e. prizes are more often given to two or three laureates instead of just one.")
                                ],
                        ),

                        # PLOT: Pie Chart: Gender
                        generate_plot_in_layout(   
                            header = "Overall Gender Distribution",
                            subheader = "The labels female and male are taken directly from the official Nobel Prize Outreach API. It would be interesting to learn how they get/set those values, or if they are simply based on perception. In any case, the cases where perception differs from self-identification may exist, but they will not substantially change the findings.",
                            plot_id = "fig_donut_gender2",
                            figure = fig_donut_gender,
                            # style={'width': '50vw', 'height': '50vh'}
                            footer = [
                                dcc.Markdown("**Interesting Findings**: This plain old pie chart reaffirms what we found already above.")
                                ],
                        ),

                        # PLOT: Pie Chart: Ethnicity
                        generate_plot_in_layout(   
                            header = "Overall Ethnicity Distribution",
                            subheader = "This plot shows the distribution of ethnicities among Nobel Laureates. I am aware that notions of ethnicity or even race can be considered problematic. There are some who suggest to not use these categorizations at all. However, I think we may loose analytical power if we do; this chart is the successor to an earlier one that showed that there are exactly zero Black Nobel laureates in the natural sciences. This certainly is an interesing finding, how ever one may interpret it.",
                            plot_id = "fig_donut_ethnicity2",
                            figure = fig_donut_ethnicity,
                            # style={'width': '50vw', 'height': '50vh'}
                            footer = [
                                dcc.Markdown("**Assignment Process**: For additional transparency, here is how I have assigned the labels. Feel free to constructively critizice it. First, I started by geography: Everyone born in Europe was assigned *European*. As a starting point, everyone born in the USA or Canada was also assigned *European*. Similarly for all other continents. That process so far already raises difficult questions as to what ethnicity is, exactly. There is a myriad of publications on this topic, so my working definition was: Where someone's family originated from, going back to before Columbus. That then introduces two new categories for North America: *African-American*, and *North American*. Why not native American? Because even the native people of almost every country immigrated at some point in human history, as far as we know. Consequently, we then have *South American*, and then again *European* for all the (mostly) Spanish and Portuguese immigrants to South America. You may miss some categories like Central America, American Indians, Alaska Natives, etc - but there are simply no laureates in these ethnicities yet, so no need for further distinction. Israel is a special case: geographically, one would have to attribute *Asian*, but historically, most Israeli (laureates) have migrated there from parts of Europe. This is also an example for the next step (after categorization by continent), where I checked various lists available on the internet (mostly Wikipedia), such as \"List of Black Nobel Laureates\", \"List of Latin American Nobel Laureates\", and so on. Whenever appropriate, I changed the label. Next, I went through all the names one by one. Due to my former occupation, I know about 60 percent of them and also know the basics of their biographies. For the remaining ones, I checked their Wikipedia pages. You may note that there is also the category *Various*, which is a more subtle version of \"Mixed\". If it said, for example, on a laureate's Wikipedia page, that he had a British father and Korean mother, then I assigned *Various*.")
                                ],
                        ),

                        # PLOT: Pie Chart: Religion
                        generate_plot_in_layout(   
                            header = "Overall Religion Distribution",
                            subheader = "This plot shows the distribution of religion among Nobel Laureates.",
                            plot_id = "fig_donut_religion2",
                            figure = fig_donut_religion,
                            # style={'width': '50vw', 'height': '50vh'}
                            footer = [
                                html.Div(dcc.Markdown(["**Note**: Yet another sightly problematic categorization, for various reasons. One of them is data availability. There are lists on Wikipedia for Jewish, Muslim and Christian laureates, which I used. My suspicion here is though that the list of Jewish laureates is more or less complete, while that of Muslim laureates is not. For the list of Christian laureates, it states that it only lists laureates that have professed their faith. So this graph is actually somewhat misleading: First of all, it is unclear wether it is about \"firm faith\" or just religious upbringing. Second, we know little about what laureates really believe, which may be different from their religion. In any case: If we were to look at religion as stated in some official documents, then I suppose the number for Muslims should be higher, the number for Christians should be much higher (close to all of European Ethnicity), and we also have to add those religions completely lacking at the moment, e.g. Asian religions (and others)."])),
                                html.Div(dcc.Markdown(["**Interesting Findings**: Even with the necessary changes described above, there still is a large number of Jewish Nobel laureates: 17.6 percent. According to Wikipedia, the Jewish religion has share among all religions in the world of 0.2 percent."]))                                ],
                        ),

                    ],
                    gap="sm"
                )
            ],
            shadow="md",
            radius="md",
            p="lg", 
            className="mt-3",
        )
    else:
        return html.Div("No content available")






##################################################################################################
# Tab Time
##################################################################################################
@app.callback(
    Output('tab-content-time', 'children'),
    Input('tabs', 'value')
)
def render_tab_time_content(active_tab):
    if active_tab == 'tab_time':
        with open('pcp_plots.pkl', 'rb') as f5:
            pcp_plots = pickle.load(f5)
        
        # Extract the figures from the loaded pickle data
        fig_histogram_timegap = pcp_plots['fig_histogram_timegap']
        fig_scatter_timegaptrend = pcp_plots['fig_scatter_timegaptrend']
        fig_scatterbox_age = pcp_plots['fig_scatterbox_age']
        fig_heatmap_age = pcp_plots['fig_heatmap_age']

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                # html.H5("Tab 4: Full Data", className="card-title"),
                html.Div("Plots on this tabe relate to the age of Nobel Laureates, as well as to the time gap between their invention/research/seminal paper and the Nobel Prize."),
                dmc.Space(h="xl"),

                dmc.Stack(
                    children=[

                        # PLOT: Histogram Timegap
                        generate_plot_in_layout(   
                            header = "Timegap Between Discovery and Prize (Histogram)",
                            subheader = "This histogram shows how often a value appears. For example, a waiting time of 11 years happened most often (=highest bar)",
                            datato = "1914/2023",
                            badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
                            plot_id = "fig_histogram_timegap",
                            figure = fig_histogram_timegap,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: Most scientists who got the Nobel prize had to wait between 1 and 30 years, with a peak around 11 years. Very long waiting times don't appear very often.")
                                ],
                            # code = dmc.Group(
                            #     children=[
                            #         html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                            #         dcc.Dropdown(
                            #             id='fig_histogram_timegap_datasource_dropdown',
                            #             options=[
                            #                 {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                            #                 {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                            #                 {'label': '1901 - 2023 (both)', 'value': 'both'}
                            #             ],
                            #             value='both',  # Default value
                            #             clearable=False,
                            #             style={"width": "300px"}
                            #         ),
                            #     ],
                            #     gap="md",  # Adjusts the space between the label and the dropdown
                            #     align="flex-start",  # Align items to the left
                            #     style={"margin-top": "20px"}
                            # )


                            # code = dmc.Stack(
                            #     children=[
                            #         html.Div("Select Data Source"),
                            #         dcc.Dropdown(
                            #             id='fig_histogram_timegap_datasource_dropdown',
                            #             options=[
                            #                 {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                            #                 {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                            #                 {'label': '1901 - 2023 (both)', 'value': 'both'}
                            #             ],
                            #             value='both',  # Default value
                            #             clearable=False,
                            #             style={"width": "300px"}
                            #         ),
                            #     ],
                            #     # gap="md",  # Adjusts the space between the label and the dropdown
                            #     # align="flex-start",  # Align items to the left
                            #     # style={"margin-top": "20px"}
                            # )

                            code = dmc.Group(
                                children=[
                                    html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_scatter_timegap_trend_datasource_dropdown',
                                        options=[
                                            {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                            {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                            {'label': '1901 - 2023 (both)', 'value': 'both'}
                                        ],
                                        value='both',  # Default value
                                        clearable=False,
                                        style={"width": "300px"}
                                    ),
                                ],
                                gap="md",  # Adjusts the space between the label and the dropdown
                                align="flex-start",  # Align items to the left
                                style={"margin-top": "20px"}
                            ),
                            code2 = dmc.Group(
                                children=[
                                    html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_scatter_timegap_trend_datasource_dropdown',
                                        options=[
                                            {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                            {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                            {'label': '1901 - 2023 (both)', 'value': 'both'}
                                        ],
                                        value='both',  # Default value
                                        clearable=False,
                                        style={"width": "300px"}
                                    ),
                                ],
                                gap="md",  # Adjusts the space between the label and the dropdown
                                align="flex-start",  # Align items to the left
                                style={"margin-top": "20px"}
                            )
                        ),

                        # PLOT: Histogram Timegap
                        generate_plot_in_layout(   
                            header = "Timegap Between Discovery and Prize (Trendlines)",
                            subheader = "This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up).",
                            datafrom= "1994",
                            datato = "2014",
                            badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
                            plot_id = "fig_scatter_timegap_trend",
                            figure = fig_scatter_timegaptrend,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: It seems that the time gap increases in a pretty similar fashion as the life expectancy.")
                                ],
                            code = dmc.Group(
                                children=[
                                    html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_scatter_timegap_trend_datasource_dropdown',
                                        options=[
                                            {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                            {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                            {'label': '1901 - 2023 (both)', 'value': 'both'}
                                        ],
                                        value='both',  # Default value
                                        clearable=False,
                                        style={"width": "300px"}
                                    ),
                                ],
                                gap="md",  # Adjusts the space between the label and the dropdown
                                align="flex-start",  # Align items to the left
                                style={"margin-top": "20px"}
                            ),
                            code2 = dmc.Group(
                                children=[
                                    html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_scatter_timegap_trend_datasource_dropdown',
                                        options=[
                                            {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                            {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                            {'label': '1901 - 2023 (both)', 'value': 'both'}
                                        ],
                                        value='both',  # Default value
                                        clearable=False,
                                        style={"width": "300px"}
                                    ),
                                ],
                                gap="md",  # Adjusts the space between the label and the dropdown
                                align="flex-start",  # Align items to the left
                                style={"margin-top": "20px"}
                            )

                        ),

                        # PLOT: Age at Award Scatter Trend
                        generate_plot_in_layout(   
                            header = "Laureate Age at Time of Award",
                            subheader = "This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up).",
                            plot_id = "fig_scatterbox_age",
                            figure = fig_scatterbox_age,
                            footer = [dcc.Markdown("**Interesting Findings**: In the early years, the average age in the natural sciences was around 45, whereas nowadays it is close to 65. This fits well to the earlier finding that the timegap has increased by - on average - 25 years. Interestingly enough, peace prize awardees get younger.")],
                        ),

                        # PLOT: Age at Award Heatmap
                        generate_plot_in_layout(   
                            header = "Laureate Age at Time of Award (Heatmap)",
                            subheader = "Same data as above, but displayed as heatmap.",
                            plot_id = "fig_heatmap_age",
                            figure = fig_heatmap_age,
                        ),

                    ],
                    gap="sm"
                )

            ],
            shadow="md",
            radius="md",
            p="lg", 
            className="mt-3",
        )
    else:
        return html.Div("No content available")



##################################################################################################
# Tab Misc
##################################################################################################
@app.callback(
    Output('tab-content-misc', 'children'),
    Input('tabs', 'value')
)
def render_tab_misc_content(active_tab):
    if active_tab == 'tab_misc':
        with open('pcp_plots.pkl', 'rb') as f5:
            pcp_plots = pickle.load(f5)
        
        # Extract the figures from the loaded pickle data
        fig_line_prizemoney = pcp_plots['fig_line_prizemoney']
        totalprizeamount = pcp_plots['totalprizeamount']
        

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                html.Div("Everything that didn't fit elsewhere."),
                dmc.Space(h="xl"),

                dmc.Stack(
                    children=[

                        # PLOT: Nobel Fields
                        generate_png_in_layout(   
                            header = "Nobel Fields",
                            subheader = "Cube sizes represent the number of Nobel prizes awarded to that field.",
                            filepath = "/assets/images/fig_cubes_fields.png",
                            style = {'height':'80vh'},
                            footer = [
                                dcc.Markdown("**Interesting Findings**: Researchers with a momentum strategy should focus their research on particle physics or immunology, while contrarians should choose ethnology or chaos theory.")
                                ],
                        ),

                        # PLOT: Prize Money
                        generate_plot_in_layout(   
                            header = "Prize Money",
                            subheader = f"In SEK; total amount paid up until today: {(totalprizeamount * 0.088):,.0f} EUR",
                            plot_id = "fig_line_prizemoney",
                            figure = fig_line_prizemoney,
                            footer = [
                                dcc.Markdown("**Interesting Findings**: Starting in the mid-1980s, the total prize amount goes up. The main reason herefore: the yearly prize amounts gets increased several times.")
                                ],
                        ),
                    ],
                    gap="sm"
                )
            ],
            shadow="md",
            radius="md",
            p="lg", 
            className="mt-3",
        )
    else:
        return html.Div("No content available")



##################################################################################################
# Tab Migration
##################################################################################################

@app.callback(
    Output('tab-content-migration', 'children'),
    Input('tabs', 'value')
)
def render_tab_migration_content(active_tab):
    if active_tab == 'tab_migration':
        with open('pcp_plots.pkl', 'rb') as f6:
            pcp_plots = pickle.load(f6)
        
        # Extract the figures from the loaded pickle data
        fig_parcat_migration_dwp = pcp_plots['fig_parcat_migration_dwp']
        fig_parcat_migration_bpd = pcp_plots['fig_parcat_migration_bpd']
        fig_globe_movement = pcp_plots['fig_globe_movement']

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                # html.H5("Tab 4: Full Data", className="card-title"),
                html.Div("This tab contains plots related to the migration of Nobel laureates."),
                dmc.Space(h="xl"),

                dmc.Stack(
                    children=[

                        # PLOT: Movement DWP
                        generate_plot_in_layout(   
                            header = "Movement: Place of Main Degree / Main Discovery / Prize",
                            subheader = [
                                dcc.Markdown("This plot shows the movement between three locations: where did the laureates get their main university degree (or similar), where did they do their main work that led to the discovery, and where did they work at the time when they received the prize? This plot is based on the Nature paper \"At what institutions did Nobel laureates do their prize-winning work?\" (see References), which unfortunately only covers the years 1994 - 2014."),
                                dcc.Markdown("**How to Read:**: The three vertical pillars stand for the three points and places in time: **degree, work, prize**. The lines show the flow from one place to the next. The on-hover infobox also shows you the overall percentage of the selected group. If you like, you may also re-arrange the bar sections via drag and drop. The dropdowns let you choose between *City* (many), *Country* (less), and the combination of both, which distinguishes Cambridge UK from Cambridge USA (etc.)")
                            ],
                            badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
                            plot_id = "fig_parcat_migration_dwp",
                            figure = fig_parcat_migration_dwp,
                            style= {'width':'80vw', 'height':'80vh'},
                            footer = [
                                dcc.Markdown("**Interesting Findings**: There are many findings to be made: For example, US laureates tend to be very immobile; however, not as immobile as the French. German researchers, on the other hand, love to go abroad - however you may want to interpret that. Finally, it is an interesting exercise to speculate if the period 1994-2014 is significantly different from other periods.")
                                ],
                            code = dmc.Group(
                                children=[
                                    html.Div("Please select location types.", style={"margin-top": "5px", "font-weight": "bold"}),
                                    html.Div("Degree", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_dwp_degree_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'ParCatDegreeCity'},
                                            {'label': 'Country', 'value': 'ParCatDegreeCountry'},
                                            {'label': 'City+Country', 'value': 'ParCatDegreeCityCountry'}
                                        ],
                                        value='ParCatDegreeCountry',  # Default value
                                        clearable=False,
                                        style={"width": "200px"} 
                                    ),
                                    html.Div("Work", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_dwp_work_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'ParCatWorkCity'},
                                            {'label': 'Country', 'value': 'ParCatWorkCountry'},
                                            {'label': 'City+Country', 'value': 'ParCatWorkCityCountry'}
                                        ],
                                        value='ParCatWorkCountry',  # Default value
                                        clearable=False,
                                        style={"width": "200px"} 
                                    ),
                                    html.Div("Prize", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_dwp_prize_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'ParCatPrizeCity'},
                                            {'label': 'Country', 'value': 'ParCatPrizeCountry'},
                                            {'label': 'City+Country', 'value': 'ParCatPrizeCityCountry'}
                                        ],
                                        value='ParCatPrizeCountry',  # Default value
                                        clearable=False,
                                        style={"width": "200px"} 
                                    ),
                                ],
                            ),
                        ),


                        # PLOT: Movement BPD
                        generate_plot_in_layout(   
                            header = "Movement: Birth / Prize / Death",
                            subheader = "This plot works the same way, but has slightly diffferent data: place of birth, place of organisation when the prize was awarded, place of death. Note that this dataset, unlike the previous one, spans the full time range. (Selecting *City* may lead to incorrect visuals, as there are simply too many to display.)",
                            badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
                            plot_id = "fig_parcat_migration_bpd",
                            figure = fig_parcat_migration_bpd,
                            #style= {'width':'1000px', 'height':'auto'},
                            footer = [
                                dcc.Markdown("**Interesting Findings**: There are many findings to be made: For example, US laureates tend to be very immobile; however, not as immobile as the French. German researchers, on the other hand, love to go abroad - however you may want to interpret that. Finally, it is an interesting exercise to speculate if the period 1994-2014 is significantly different from other periods.")
                                ],
                            code = dmc.Group(
                                children=[
                                    html.Div("Please select location types.", style={"margin-top": "5px", "font-weight": "bold"}),
                                    html.Div("Degree", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_bpd_birth_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'BirthCityNow'},
                                            {'label': 'Country', 'value': 'BirthCountryNow'},
                                            {'label': 'Continent', 'value': 'BirthContinent'},
                                        ],
                                        value='BirthCountryNow',  # Default value
                                        clearable=False,
                                        style={"width": "200px"}  
                                    ),
                                    html.Div("Work", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_bpd_prize_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'Prize0_Affiliation0_CityNow'},
                                            {'label': 'Country', 'value': 'Prize0_Affiliation0_Country'},
                                            {'label': 'Continent', 'value': 'Prize0_Affiliation0_Continent'}
                                        ],
                                        value='Prize0_Affiliation0_Country',  # Default value
                                        clearable=False,
                                        style={"width": "200px"}  
                                    ),
                                    html.Div("Prize", style={"margin-top": "5px", "font-weight": "bold"}),
                                    dcc.Dropdown(
                                        id='fig_parcat_migration_bpd_death_dropdown',
                                        options=[
                                            {'label': 'City', 'value': 'DeathCityNow'},
                                            {'label': 'Country', 'value': 'DeathCountryNow'},
                                            {'label': 'Continent', 'value': 'DeathContinent'}
                                        ],
                                        value='DeathCountryNow',  # Default value
                                        clearable=False,
                                        style={"width": "200px"} 
                                    ),
                                ],
                            ),
                        ),




                        # PLOT: Globe Movement
                        generate_plot_in_layout(   
                            header = "Movement: Birth / Prize",
                            subheader = "This globe shows the movement from place of birth to place of affiliation at the time of the award.)",
                            badges = [dmc.Badge("All Categories", variant="outline", color=brand_color_alt)],
                            plot_id = "fig_globe_movement",
                            figure = fig_globe_movement,
                            #style= {'width':'1000px', 'height':'auto'},
                            footer = [
                                dcc.Markdown("**Interesting Findings**: Isn't it nice to look at?")
                                ],
                        )

                    ],
                    gap="sm"
                )
            ],
            shadow="md",
            radius="md",
            p="lg", 
            className="mt-3",
        )
    else:
        return html.Div("No content available")

##################################################################################################
# Tab Data
##################################################################################################
@app.callback(
    Output('tab-content-data', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == 'tab_data':

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                html.H3("About"),
                html.Div(dcc.Markdown(["The Nobel Laureate Data Dashboard is a project of Wolfgang Huang. If you want to learn more about my other projects, please see my portfolio page www.virtuousvector.ai ([Link](https://www.virtuousvector.ai)), or email me at mail-at-virtuousvector.ai."])),
                dmc.Space(h="xl"),
                
                html.H3("Data"),

                html.H5("Nobel Laureate Base Data"),
                html.Div("The core of the data is provided by Nobel Prize Outreach via their API. You can view it below, or access it via the API yourself. By the way, you can also sort and filter the data by clicking on the column headers / the column burger menu."),
                dmc.Space(h="xl"),
                html.Div([ag_df_laureates]),

                dmc.Space(h="xl"),

                html.H5("Timegap Analysis"),
                html.Div("The data used for timegap analysis is published here:"),
                html.Div(dcc.Markdown(["Li, Jichao; Yin, Yian; Fortunato, Santo; Wang Dashun, 2018, \"A dataset of publication records for Nobel laureates\", Harvard Dataverse, [Link](https://doi.org/10.7910/DVN/6NJ5RN)"])),

                dmc.Space(h="xl"),

                html.H5("Degree - Work - Prize Migration Analysis"),
                html.Div("The data used for migration analysis is published here:"),
                html.Div(dcc.Markdown(["Schlagberger, E.M., Bornmann, L. & Bauer, J.: \"At what institutions did Nobel laureates do their prize-winning work? An analysis of biographical information on Nobel laureates from 1994 to 2014\". Scientometrics 109, 723–767 (2016). [Link](https://doi.org/10.1007/s11192-016-2059-2)"])),

                dmc.Space(h="xl"),

                html.H5("Population & Life Expectancy"),
                html.Div("The data used for population numbers and life expectancy is published here:"),
                html.Div(dcc.Markdown(["Gapminder.org Data Downloads [Link](https://www.gapminder.org/data/)"])),


                dmc.Space(h="xl"),

                html.H5("Ethnicity"),
                html.Div("The data used for ethnicity is self-compiled. Further details are provided alongside the plot. Feel free to contact me if you have constructive criticism."),
                dmc.Space(h="xl"),
                html.Div([ag_df_ethnicity]),

                dmc.Space(h="xl"),

                html.H5("Religion"),
                html.Div("The data used for religion is self-compiled. Further details are provided alongside the plot. Feel free to contact me if you have constructive criticism."),
                dmc.Space(h="xl"),
                html.Div([ag_df_religion]),

                dmc.Space(h="xl"),

                html.H5("Download the Data"),
                html.Div(dcc.Markdown(["You can download all the data from my Github repository \"nobeldashboard\". [Link](https://github.com/WolfgangHuang/nobeldashboard)"])),
                dmc.Space(h="xl"),
                
            ],
            shadow="md",
            radius="md",
            p="lg",  
            className="mt-3",
        )
    else:
        return html.Div("No content available")





##################################################################################################
# Callbacks for User Interaction
##################################################################################################


# Nationality Globe
@app.callback(
    Output('fig_choroplethglobe_prizespercountry', 'figure'),
    [Input('fig_choroplethglobe_year_slider', 'value')]
)
def update_globe(year_range):
    return pcp.generate_choroplethglobe(year_range=year_range, year_range_field="award")

# Cities of Birth
@app.callback(
    Output('fig_map_cities', 'figure'),
    Input('fig_map_cities_city_dropdown', 'value'),
)
def update_cities_map(selected_city_type): # the passed value here is passed before from the callback automatically
    return pcp.generate_map_cities(city=selected_city_type)


# Timegap Histogram
@app.callback(
    Output('fig_histogram_timegap', 'figure'),
    Input('fig_histogram_timegap_datasource_dropdown', 'value'),
)
def update_timegap_plot(selected_datasource): # the passed value here is passed before from the callback automatically
    return pcp.generate_histogram_timegap(datasource=selected_datasource)


# Timegap Scatterbox
@app.callback(
    Output('fig_scatter_timegap_trend', 'figure'),
    Input('fig_scatter_timegap_trend_datasource_dropdown2', 'value'),
)
def update_timegap_trend_plot(selected_datasource): # the passed value here is passed before from the callback automatically
    return pcp.generate_timegap_trend(datasource=selected_datasource)


# DWP Migration DWP
@app.callback(
    Output('fig_parcat_migration_dwp', 'figure'),
    Input('fig_parcat_migration_dwp_degree_dropdown', 'value'),
    Input('fig_parcat_migration_dwp_work_dropdown', 'value'),
    Input('fig_parcat_migration_dwp_prize_dropdown', 'value'),
)
def update_migration_parcat_1(loc1, loc2, loc3): # the passed value here is passed before from the callback automatically
    return pcp.generate_parcat_migration(loc1=loc1, loc2=loc2, loc3=loc3)


# DWP Migration BPD
@app.callback(
    Output('fig_parcat_migration_bpd', 'figure'),
    Input('fig_parcat_migration_bpd_birth_dropdown', 'value'),
    Input('fig_parcat_migration_bpd_prize_dropdown', 'value'),
    Input('fig_parcat_migration_bpd_death_dropdown', 'value'),
)
def update_migration_parcat_2(loc1, loc2, loc3): # the passed value here is passed before from the callback automatically
    return pcp.generate_parcat_migration(loc1=loc1, loc2=loc2, loc3=loc3)



##################################################################################################
# Running the app
##################################################################################################

# --------------------
# Run the app (locally)
if __name__ == "__main__":
    app.run(debug=True, port=5085)

# # # Run the app on the server
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8050))  # Fallback to port 8050 if PORT isn't set
#     app.run_server(host='0.0.0.0', port=port, debug=False)
