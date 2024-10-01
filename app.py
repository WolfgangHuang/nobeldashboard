##################################################################################################
# Library Imports
##################################################################################################


import plotly.express as px
import plotly.graph_objects as go
import plotly
import pandas as pd
import dash_ag_grid as dag
import os
import json
import numpy as np
import dash
from dash import dcc, html
from dash import Dash, _dash_renderer
from dash.dependencies import Input, Output
import dash_mantine_components as dmc
from sklearn.linear_model import LinearRegression
import pickle
import precompute_plots as pcp 


##################################################################################################
# General Settings
##################################################################################################

brand_color_plot_background='#FAFAFA'

c_brown = '#47382a'
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


colorscale_palette = [c_teal, c_lightblue, c_red, c_orange, c_yellow, c_darkmagenta, c_magenta, c_pink]
colorscale_red = [c_red_verylight, c_red_verydark]
colorscale_teal = [c_teal_verylight, c_teal_verydark]
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

# Load precomputed plots from file
with open('precomputed_plots.pkl', 'rb') as f:
    precomputed_plots = pickle.load(f)

fig_1b = precomputed_plots['fig_1b']
fig_1c = precomputed_plots['fig_1c']
fig_5a = precomputed_plots['fig_5a']
fig_5b = precomputed_plots['fig_5b']
fig_5c = precomputed_plots['fig_5c']
fig_2a = precomputed_plots['fig_2a']
fig_2b = precomputed_plots['fig_2b']
fig_2c = precomputed_plots['fig_2c']
fig_2d = precomputed_plots['fig_2d']
fig_2e = precomputed_plots['fig_2e']
fig_3a = precomputed_plots['fig_3a']
fig_3b = precomputed_plots['fig_3b']
fig_3c = precomputed_plots['fig_3c']
fig_3d = precomputed_plots['fig_3d']
fig_4a = precomputed_plots['fig_4a']
fig_4b = precomputed_plots['fig_4b']
fig_6a = precomputed_plots['fig_6a']


# Load tables

# Laureates data
df_laureates = pd.read_csv('df_laureates_cleaned.csv', sep=';', encoding="UTF-8")

# Same as laureates, but the two-time-winners are listed twice
df_prizes = pd.read_csv('df_prizes_cleaned.csv', sep=';', encoding="UTF-8")

# Timegap Seminal Paper and Prize
df_timegap = pd.read_csv('df_prize-publication-timegap.csv', sep=';', encoding='UTF-8')

# Average life expectancy data
df_lifeexpectancy = pd.read_excel('df_life-expectancy.xlsx')

# Country Populations
df_pop = pd.read_excel("df_population.xlsx")

# Ethnicity
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';', encoding="UTF-8")

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';', encoding="UTF-8")

# Degree - Work - Prize Movement
df_movement_dwp = pd.read_excel('df_degree_institutions_work.xlsx')
df_movement_dwp = df_movement_dwp.fillna('None')

# Birth - Prize - Death Movement
df_movement_bpd = df_laureates[["BirthCityNow", "BirthCountryNow","BirthContinent","Prize0_Affiliation0_CityNow", "Prize0_Affiliation0_Country","Prize0_Affiliation0_Continent", "DeathCityNow", "DeathCountryNow", "DeathContinent"]]
df_movement_bpd = df_movement_bpd.fillna('None')

# ISO3 list
df_iso = pd.read_csv('countries_iso2_iso3.csv', sep=';', encoding="UTF-8")

# Nobelprizes per Country
df_nobelprizes_percountry = pd.read_csv("df_nobelprizes_percountry.csv", sep=';', encoding="UTF-8")

max_prize_count = df_nobelprizes_percountry['Count'].max()


##################################################################################################
# Dashboard Main Setup
##################################################################################################

_dash_renderer._set_react_version("18.2.0")

app = Dash(
    external_stylesheets=[
        "assets/dmc_styles.css",  # Your custom CSS
        dmc.styles.ALL           # Mantine styles
    ],
    title="Nobel Laureate Data Dashboard"
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

##################################################################################################
# Tab Contents
##################################################################################################

# Contents of Tab 0

tab0_content = dmc.Paper(
    children=[
        dmc.Group(
            children=[

                html.Div(
                    children=[
                            html.Div(
                                "1000",
                                className="large-number"
                            ),
                            html.Div(
                                "Exactly 1000 prizes have been awarded since 1901. Of these, two were declined, and one not received.",
                                className="description"
                            )
                    ],
                    className="info-box"
                ),

                html.Div(
                    children=[
                            html.Div(
                                "18",
                                className="large-number"
                            ),
                            html.Div(
                                "Up to 18 new laureates could be announced 2024, three per category. In contrast, prizes could also be given to only 6 recipients.",
                                className="description"
                            )
                    ],
                    className="info-box"
                ),

               
                html.Div(
                    children=[
                            html.Div(
                                f"{(pcp.totalprizeamount*0.088):,.0f}",
                                className="small-number"
                            ),
                            html.Div(
                                "Inflation-adjusted total prize amount paid until today in EUR.",
                                className="description"
                            )
                    ],
                    className="info-box"
                )

            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Space(h="xl")]),
        
        dmc.Group(
            children=[

                dmc.Grid(
                    children=[
                        dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_6a', figure=fig_6a, style={'width': '800px', 'height': '400px'})), span=8)
                    ]
                )
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Space(h="xl")]),

        dmc.Group(
            children=[

                html.Div(dcc.Loading(dcc.Graph(id='fig_2c', figure=fig_2c)), style={'width': '265px', 'justify':'left'}),
                html.Div(dcc.Loading(dcc.Graph(id='fig_2d', figure=fig_2d)), style={'width': '265px', 'justify':'left'}),
                html.Div(dcc.Loading(dcc.Graph(id='fig_2e', figure=fig_2e)), style={'width': '265px', 'justify':'left'})

            ],             
        )           
    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)

# Contents of Tab 1

tab1_content = dmc.Paper(
    children=[
        html.Div("This tab contains plots related to the nationality of Nobel Laureates. Actually, most of the time, it refers to the country of birth, which is unique and easy to identify, unlike the actual nationality. Note that quite some Laureates will actually not have the nationality of the country they were born in."),
        dmc.Space(h="xl"),
        html.H2("Nobel Prizes by Country of Birth [1b]"),
        html.Div(dcc.Markdown(["This plot shows the distribution of country of birth of the laureates; you may rotate the globe, and zoom in and out. The slider lets you select minimum and maximum number of laureates, e.g.\"*less than 5 laureates*\"."])),
        dmc.Space(h="xl"),
        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),
        dmc.Space(h="xl"),
        dmc.Grid(
            children=[
                dmc.GridCol(
                    dcc.RangeSlider(
                        id='prize-slider',
                        min=0,
                        max=max_prize_count,  # Dynamic maximum value based on data
                        step=1,
                        value=[0, max_prize_count],  # Default range from 0 to max
                        marks={i: str(i) for i in range(0, int(max_prize_count) + 1, 50)}, 
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="dmc-bar dmc-thumb",
                    ),
                    span=5
                ),
            ],
            justify="left",
            style={"margin-top": "30px"}
        ),
        dmc.Space(h="xl"),
        dmc.Grid(
            html.Div(
                children=[
                    dmc.GridCol(
                        dcc.Loading(
                            dcc.Graph(
                                id='fig_1b',
                                figure=fig_1b,
                                config={'responsive': True},
                                style={'width': '70vw', 'height': '80vh'}
                            )
                        ),
                        span=12,
                    )
                ],
            )
        ),

               
        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Places of Birth and Death [1c]"),

        dmc.Space(h="xl"),
        html.Div("This map shows the cities of birth and death of the laureates. Note that the points of the map are given as center points of the respective cities, not as the actual places of birth (e.g. hospitals). You can zoom in to quite some detail; the map data is provided via OpenStreetMap. "),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Group(
            children=[
                html.Div("Please select location:", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='city-dropdown',
                    options=[
                        {'label': 'City of Birth', 'value': 'birth'},
                        {'label': 'City of Death', 'value': 'death'}
                    ],
                    value='birth',  # Default value
                    clearable=False,
                    style={"width": "200px"}
                ),
            ],
            gap="md",  # Adjusts the space between the label and the dropdown
            align="flex-start",  # Align items to the left
            style={"margin-top": "20px"}
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(
                    dcc.Loading(
                        dcc.Graph(
                            id='fig_1c',
                            figure=fig_1c,
                            style={'width': '80vw', 'height': '80vh'}
                        )
                    ), 
                    span=12
                )
            ]
        ),
        

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Nobel Prizes by Country of Birth and Population [5a]"),

        dmc.Space(h="xl"),
        html.Div("This plot shows the number of prizes by country of birth, but in relation to the population size of the country in the respective year. You can use the slider or play button to see the animation of the years."),
        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["**How to Read**: The actual number of prizes is shown as a number after the country name. The size of the bubble relates to the population size, but is adjusted to make smaller populations appear bigger, and bigger populations smaller; otherwise, China and India would overlap everything else. The x-axis shows the population on a log scale, again as otherwise the large countries would push the small countries to the far left edge. The y-axis shows the number of prizes per 1 million inhabitants. The scale is ajusted to the 4th root of that value, which makes the range/visible area from 0-1 very large, and that from 10-20 relatively small. Otherwise, the tiny countries with one or two laureates would push the rest to the bottom."])),
        html.Div(dcc.Markdown(["**Interesting Findings**: The visualization shows, for exampe, that countries like the USA only start to play an important role after World War II; also, in relation to their population number, they did not get unusually many Nobel Prizes. Small population numbers and a few prizes bring you to the top of the chart, as St. Lucia and Iceland show. On the other hand, large countries like India and China still have a bad prize-population ratio, partly simply due to their large population."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_5a', figure=fig_5a, style={'width': '80vw', 'height': '80vh'})), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Nobel Prizes by Country of Birth per Year [5b]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot shows the number of prizes per country of birth of laureates by year"])),
        html.Div(dcc.Markdown(["**Interesting Findings**: The many pink lines on top in the right part of the plot emphasize our earlier finding that the number of prizes given to the USA has increased tremendously only after World War II."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_5b', figure=fig_5b, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Nobel Prizes by Country of Birth per Year (Running Sum) [5c]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot shows the running sum of prizes per country of birth of laureates by year."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_5c', figure=fig_5c, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),
    ],

    shadow="md",
    radius="md",
    p="lg",
    className="mt-3",
)


# Contents of Tab 2

tab2_content = dmc.Paper(
    children=[
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div(dcc.Markdown(["This tab contains plots related to gender, ethnicity and religion."])),
        dmc.Space(h="xl"),

        html.H2("Nobel Prizes Awarded to Women [2a]"),

        html.Div(dcc.Markdown(["This plot shows the number of prizes for women in all of the disciplines per decade. It allows you to see when and in which disciplines the most prizes were awarded to women. Feel free to rotate the plot and zoom."])),

        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_2a', figure=fig_2a, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),
    

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Nobel Prizes Awarded to Men & Women [2b]"),

        html.Div(dcc.Markdown(["This plot is identical to the above, but here, men and women are both shown as two surfaces."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_2b', figure=fig_2b, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),


        html.H2("Gender Distribution [2c]"),

        html.Div(dcc.Markdown(["If the 3D surfaces above were to difficult to interpret, here is a plain old pie chart. :-)"])),
        html.Div(dcc.Markdown(["**Note**: The labels *female* and *male* are taken directly from the official Nobel Prize Outreach API. It would be interesting to learn how they get/set those values, or if they are simply based on perception. In any case, the cases where perception differs from self-identification may exist, but they will not substantially change the findings."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_2c2', figure=fig_2c, style={'width': '50vw', 'height': '50vh'})), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),


        html.H2("Ethnicity Distribution [2d]"),

        html.Div(dcc.Markdown(["This plot shows the distribution of ethnicities among Nobel Laureates."])),
        html.Div(dcc.Markdown(["**Note**: I am aware that notions of ethnicity or even race can be considered problematic. There are some who suggest to not use these categorizations at all. However, I think we may loose analytical power if we do; this chart is the successor to an earlier one that showed that there are exactly zero Black Nobel laureates in the natural sciences. This certainly is an interesing finding, how ever one may interpret it."])),
        html.Div(dcc.Markdown(["**Assignment Process**: For additional transparency, here is how I have assigned the labels. Feel free to constructively critizice it. First, I started by geography: Everyone born in Europe was assigned *European*. As a starting point, everyone born in the USA or Canada was also assigned *European*. Similarly for all other continents. That process so far already raises difficult questions as to what ethnicity is, exactly. There is a myriad of publications on this topic, so my working definition was: Where someone's family originated from, going back to before Columbus. That then introduces two new categories for North America: *African-American*, and *North American*. Why not native American? Because even the native people of almost every country immigrated at some point in human history, as far as we know. Consequently, we then have *South American*, and then again *European* for all the (mostly) Spanish and Portuguese immigrants to South America. You may miss some categories like Central America, American Indians, Alaska Natives, etc - but there are simply no laureates in these ethnicities yet, so no need for further distinction. Israel is a special case: geographically, one would have to attribute *Asian*, but historically, most Israeli (laureates) have migrated there from parts of Europe. This is also an example for the next step (after categorization by continent), where I checked various lists available on the internet (mostly Wikipedia), such as \"List of Black Nobel Laureates\", \"List of Latin American Nobel Laureates\", and so on. Whenever appropriate, I changed the label. Next, I went through all the names one by one. Due to my former occupation, I know about 60 percent of them and also know the basics of their biographies. For the remaining ones, I checked their Wikipedia pages. You may note that there is also the category *Various*, which is a more subtle version of \"Mixed\". If it said, for example, on a laureate's Wikipedia page, that he had a British father and Korean mother, then I assigned *Various*."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_2d2', figure=fig_2d, style={'width': '50vw', 'height': '50vh'})), span=12)
            ]
        ),


        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),


        html.H2("Religion Distribution"),

        html.Div(dcc.Markdown(["This plot shows the distribution of religion among Nobel Laureates."])),
        html.Div(dcc.Markdown(["**Note**: Yet another sightly problematic categorization, for various reasons. One of them is data availability. There are lists on Wikipedia for Jewish, Muslim and Christian laureates, which I used. My suspicion here is though that the list of Jewish laureates is more or less complete, while that of Muslim laureates is not. For the list of Christian laureates, it states that it only lists laureates that have professed their faith. So this graph is actually somewhat misleading: First of all, it is unclear wether it is about \"firm faith\" or just religious upbringing. Second, we know little about what laureates really believe, which may be different from their religion. In any case: If we were to look at religion as stated in some official documents, then I suppose the number for Muslims should be higher, the number for Christians should be much higher (close to all of European Ethnicity), and we also have to add those religions completely lacking at the moment, e.g. Asian religions (and others)."])),
        html.Div(dcc.Markdown(["**Interesting Findings**: Even with the necessary changes described above, there still is a large number of Jewish Nobel laureates: 17.6 percent. According to Wikipedia, the Jewish religion has share among all religions in the world of 0.2 percent."])),

        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_2e2', figure=fig_2e, style={'width': '50vw', 'height': '50vh'})), span=12)
            ]
        )
    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)

# Contents of Tab 3

tab3_content = dmc.Paper(
    children=[
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("Plots on this tabe relate to the age of Nobel Laureates, as well as to the time gap between their invention/research/seminal paper and the Nobel Prize."),
        dmc.Space(h="xl"),

        html.H2("Timegap Between Discovery and Prize (Histogram) [3a]"),
        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot shows the time that has passed between \"the invention/discovery\" and the awardment of the Nobel Prize. The data used for this plot is from the Nature paper \"A dataset of publication records for Nobel laureates\" (see References tab). This data spans from 1901 to 2014, with quite some ommissions in the early years, where it was not clear when the main discovery was made. As for the definition of this time point in general, please refer to the paper. In addition to that, I queried ChatGPT-o1 for the laureates from 2015 - 2023. You can use the dropdown to select which data should be displayed.  "])),
        html.Div(dcc.Markdown(["**How to Read**: The data is provided as histogram, meaning it counts how often a value appears. The x-axis reflects the time gap, and the y-axis shows how often a timegap has this value. As you can see, the most common time gap is 11 years, which has happened 14 times."])),
        html.Div(dcc.Markdown(["**Interesting Findings**: You often hear that you would have to wait decades for the prize; which is true some extent; in the majority of the cases, however, it doesn't take longer than 25 years."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2014/2023", variant="outline", color= brand_color_main),
            dmc.Badge("Natural Sciences", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Group(
            children=[
                html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='datasource_dropdown',
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

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_3a', figure=fig_3a, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        html.H2("Timegap Between Discovery and Prize (Scatterbox with Trendlines) [3b]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up)."])),
        html.Div(dcc.Markdown(["**Interesting Findings**: It seems that the time gap increases in a pretty similar fashion as the life expectancy."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1994 - 2014", variant="outline", color= brand_color_main),
            dmc.Badge("Natural Sciences", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Group(
            children=[
                html.Div("Please select data:", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='datasource_dropdown2',
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

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_3b', figure=fig_3b, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

        
        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        html.H2("Laureate Age at Time of Award (Scatterbox) [3c]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot shows the age of Nobel laureates at the time when they received the award."])),
        html.Div(dcc.Markdown(["**Interesting Findings**: In the early years, the average age in the natural sciences was around 45, whereas nowadays it is close to 65. This fits well to the earlier finding that the timegap has increased by - on average - 25 years. Interestingly enough, peace prize awardees get younger."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_3c', figure=fig_3c, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

        
        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

        html.H2("Laureate Age at Time of Award (Heatmap) [3d]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["Same data as above, but displayed as heatmap."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("All Categories", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_3d', figure=fig_3d, style={'width': '80vw', 'height': '60vh'})), span=12)
            ]
        ),

    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)


# Contents of Tab 4

tab4_content = dmc.Paper(
    children=[
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This tab contains plots related to the migration of Nobel laureates."),
        dmc.Space(h="xl"),

        html.H2("Movement: Place of Main Degree / Main Discovery / Prize [4a]"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot shows the movement between three locations: where did the laureates get their main university degree (or similar), where did they do their main work that led to the discovery, and where did they work at the time when they received the prize? This plot is based on the Nature paper \"At what institutions did Nobel laureates do their prize-winning work?\" (see References), which unfortunately only covers the years 1994 - 2014."])),
        html.Div(dcc.Markdown(["**How to Read:**: The three vertical pillars stand for the three points and places in time: **degree, work, prize**. The lines show the flow from one place to the next. The on-hover infobox also shows you the overall percentage of the selected group. If you like, you may also re-arrange the bar sections via drag and drop. The dropdowns let you choose between *City* (many), *Country* (less), and the combination of both, which distinguishes Cambridge UK from Cambridge USA (etc.)"])),
        html.Div(dcc.Markdown(["**Interesting Findings**: There are many findings to be made: For example, US laureates tend to be very immobile; however, not as immobile as the French. German researchers, on the other hand, love to go abroad - however you may want to interpret that. Finally, it is an interesting exercise to speculate if the period 1994-2014 is significantly different from other periods."])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1994 - 2014", variant="outline", color= brand_color_main),
            dmc.Badge("Natural Sciences", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Group(
            children=[
                html.Div("Please select location types.", style={"margin-top": "5px", "font-weight": "bold"}),
                html.Div("Degree", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='degree-dropdown',
                    options=[
                        {'label': 'City', 'value': 'DegreeCity'},
                        {'label': 'Country', 'value': 'DegreeCountry'},
                        {'label': 'City+Country', 'value': 'DegreeCityCountry'}
                    ],
                    value='DegreeCountry',  # Default value
                    clearable=False,
                    style={"width": "200px"} 
                ),
                html.Div("Work", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='work-dropdown',
                    options=[
                        {'label': 'City', 'value': 'WorkCity'},
                        {'label': 'Country', 'value': 'WorkCountry'},
                        {'label': 'City+Country', 'value': 'WorkCityCountry'}
                    ],
                    value='WorkCountry',  # Default value
                    clearable=False,
                    style={"width": "200px"} 
                ),
                html.Div("Prize", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='prize-dropdown',
                    options=[
                        {'label': 'City', 'value': 'PrizeCity'},
                        {'label': 'Country', 'value': 'PrizeCountry'},
                        {'label': 'City+Country', 'value': 'PrizeCityCountry'}
                    ],
                    value='PrizeCountry',  # Default value
                    clearable=False,
                    style={"width": "200px"} 
                ),
            ],
        ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_4a', figure=fig_4a, style={'width': '80vw', 'height': '95vh'})), span=12)
            ]
        ),
        
    dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="sm"), dmc.Space(h="xl")]),

    html.H2("Movement: Birth / Prize / Death"),

        dmc.Space(h="xl"),
        html.Div(dcc.Markdown(["This plot works the same way, but has slightly diffferent data: place of birth, place of organisation when the prize was awarded, place of death. Note that this dataset, unlike the previous one, spans the full time range. (Selecting *City* may lead to incorrect visuals, as there are simply too many to display.)"])),
        dmc.Space(h="xl"),

        dmc.Group(
            [
            dmc.Badge("1901 - 2023", variant="outline", color= brand_color_main),
            dmc.Badge("Natural Sciences", variant="outline", color= brand_color_alt),
            ]
        ),

        dmc.Group(
            children=[
                html.Div("Please select location types.", style={"margin-top": "5px", "font-weight": "bold"}),
                html.Div("Degree", style={"margin-top": "5px", "font-weight": "bold"}),
                dcc.Dropdown(
                    id='4b_birth_dropdown',
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
                    id='4b_prize_dropdown',
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
                    id='4b_death_dropdown',
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

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Loading(dcc.Graph(id='fig_4b', figure=fig_4b, style={'width': '80vw'})), span=12) # , 'height': '95vh'
            ]
        )



    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)





tabdata_content = dmc.Paper(
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


##################################################################################################
# Layoit Definition
##################################################################################################

app.layout = dmc.MantineProvider(
        theme={
        "colors": {
            "dmc_color1": [                
                "#B59B82",
                "#A78769",
                "#927356",
                "#795F47",
                "#604B38",
                "#47382A",
                "#403225",
                "#392C21",
                "#31271D",
                "#2A2119",               
            ],
            "dmc_color2": [                
                "#67D6E0",
                "#48CDDA",
                "#2BC4D2",
                "#25A8B3",
                "#1F8B95",
                "#186F77",
                "#16646B",
                "#13595F",
                "#114D53",
                "#0F4247",
             
            ],"dmc_color3": [                
                "#EC6570",
                "#E84653",
                "#E42737",
                "#CF1A29",
                "#B01623",
                "#91121D",
                "#83101A",
                "#740E17",
                "#660D14",
                "#570B11",             
            ],
        },
    },
    children=[
        dmc.Container(
            children=[
                dmc.Grid(
                    children=[
                        dmc.GridCol(html.H1("Nobel Laureate Data Dashboard v01.22.15", className="text-left mt-5 mb-5"), span=12)
                    ]
                ),
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("Quick Fun Facts", value="tab0"),
                                dmc.TabsTab("Nationality & Country", value="tab1"),
                                dmc.TabsTab("Gender, Ethnicity & Religion", value="tab2"),
                                dmc.TabsTab("Time & Age", value="tab3"),
                                dmc.TabsTab("Migration", value="tab4"),
                                dmc.TabsTab("Data & References", value="tabdata"),
                            ]
                        ),
                        dmc.TabsPanel(tab0_content, value="tab0"),
                        dmc.TabsPanel(tab1_content, value="tab1"),
                        dmc.TabsPanel(tab2_content, value="tab2"),
                        dmc.TabsPanel(tab3_content, value="tab3"),
                        dmc.TabsPanel(tab4_content, value="tab4"),
                        dmc.TabsPanel(tabdata_content, value="tabdata"),
                    ],
                    value="tab0",  # Default selected tab
                    id="tabs",
                ),
            ],
            fluid=True,
            style={"margin": "20px"}  # Adding 20px margin on all sides
        )
    ]
)

##################################################################################################
# Callbacks
##################################################################################################


# Nationality Globe
@app.callback(
    Output('fig_1b', 'figure'),
    [Input('prize-slider', 'value')]
)
def update_globe(selected_range):
    # Filter the data based on the selected range
    filtered_data = df_nobelprizes_percountry[
        (df_nobelprizes_percountry['Count'] >= selected_range[0]) &
        (df_nobelprizes_percountry['Count'] <= selected_range[1])
    ]
    # Generate and return the updated globe plot
    return pcp.generate_globe_plot(filtered_data)

# Cities of Birth
@app.callback(
    Output('fig_1c', 'figure'),
    Input('city-dropdown', 'value'),
)
def update_cities_map(selected_city_type): # the passed value here is passed before from the callback automatically
    return pcp.generate_scatterbox_plot(df_laureates, selected_city_type)


# Timegap Histogram
@app.callback(
    Output('fig_3a', 'figure'),
    Input('datasource_dropdown', 'value'),
)
def update_timegap_plot(selected_datasource): # the passed value here is passed before from the callback automatically

    if selected_datasource == "paper":
        filtered_data = df_timegap[(df_timegap['Source']) == "Harvard Dataverse"]

    elif selected_datasource == "chatgpt":
        filtered_data = df_timegap[(df_timegap['Source']) == "ChatGPT 4c (September 2024)"]
    else:
        filtered_data = df_timegap
    
    return pcp.generate_timegap_histogram(filtered_data)


# Timegap Scatterbox
@app.callback(
    Output('fig_3b', 'figure'),
    Input('datasource_dropdown2', 'value'),
)
def update_timegap_plot(selected_datasource): # the passed value here is passed before from the callback automatically

    if selected_datasource == "paper":
        filtered_data = df_timegap[(df_timegap['Source']) == "Harvard Dataverse"]

    elif selected_datasource == "chatgpt":
        filtered_data = df_timegap[(df_timegap['Source']) == "ChatGPT 4c (September 2024)"]
    else:
        filtered_data = df_timegap
    
    return pcp.generate_timegap_trend(filtered_data, df_lifeexpectancy)


# DWP Migration DWP
@app.callback(
    Output('fig_4a', 'figure'),
    Input('degree-dropdown', 'value'),
    Input('work-dropdown', 'value'),
    Input('prize-dropdown', 'value'),
)
def update_migration_parcat_1(loc1, loc2, loc3): # the passed value here is passed before from the callback automatically
    return pcp.generate_migration_dwp(df_movement_dwp, loc1=loc1, loc2=loc2, loc3=loc3)


# DWP Migration BPD
@app.callback(
    Output('fig_4b', 'figure'),
    Input('4b_birth_dropdown', 'value'),
    Input('4b_prize_dropdown', 'value'),
    Input('4b_death_dropdown', 'value'),
)
def update_migration_parcat_2(loc1, loc2, loc3): # the passed value here is passed before from the callback automatically
    return pcp.generate_migration_bpd(df_movement_bpd, loc1=loc1, loc2=loc2, loc3=loc3)

##################################################################################################
# Running the app
##################################################################################################

# --------------------
# Run the app (locally)
# if __name__ == "__main__":
#     app.run(debug=True, port=5085)

# Run the app on the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Fallback to port 8050 if PORT isn't set
    app.run_server(host='0.0.0.0', port=port, debug=True)

