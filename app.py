##################################################################################################
# Library Imports
##################################################################################################

import os
from dotenv import load_dotenv
import pandas as pd
import polars as pl
import dash_ag_grid as dag
import dash
from dash import dcc, html, Dash, State, callback_context  #, _dash_renderer
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, MATCH, ALL
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json
import plotdatagenerator as pdg


##################################################################################################
# List of Generator Functions and the plots generated
##################################################################################################

# choroplethglobe
# 	- fig_choroplethglobe_prizespercountry (demography)
# scattermapbox_cities
# 	- fig_map_cities (demography)
# bubbles_perpopulation
# 	fig_bubbles_population (demography)
# bar_percountry
# 	- fig_bar_prizespercountry (demography)
# 	- fig_bar_prizespercountry_rs (demography)
# 3dsurface_pergender
# 	- fig_surface_prizesforwomen (demography)
# 	- fig_surface_prizesformenwomen (demography)
# donut
# 	- fig_donut_gender (overview)
# 	- fig_donut_gender2 (demography)
# 	- fig_donut_religion (overview)
# 	- fig_donut_religion2 (demography)
# 	- fig_donut_ethnicity (overview)
# 	- fig_donut_ethnicity2 (demography)
# histogram_timegap
# 	- fig_histogram_timegap (time)
# scatterbox_timegaptrend
# 	- fig_scatter_timegaptrend (time)
# scatterbox_age
# 	- fig_scatterbox_age (time)
# heatmap_age
# 	- fig_heatmap_age (time)
# parcat_migration
# 	- fig_parcat_migration_dwp (migration)
# 	- fig_parcat_migration_bpd (migration)
# line_prizemoney
# 	- fig_line_prizemoney (misc)
# sunburst
# 	- fig_sunburst_last (current)
# globe_movemment
# 	- fig_globe_movement (migration)
# map_movement
# 	- fig_map_movement (current)
# generate_mostcommon_firstnames
#   - fig_mostcommon_firstnames (misc)


##################################################################################################
# Data Loading
##################################################################################################

# Get the path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Load tables
df_laureates = pd.read_csv('df_laureates_enriched_redux_clean.csv', sep=';', encoding="UTF-8", index_col=0)
df_prizes = pd.read_csv('df_prizes_enriched_redux_clean.csv', sep=';', encoding="UTF-8", index_col=0)
df_prizestats = pd.read_csv("df_prizestats.csv", sep=';', encoding="UTF-8")

df = pdg.count_per_country()
max_prize_count = df['Count'].max()
lastyearincluded = pdg.get_lastyearincluded()
numberofprizes = df_prizes.shape[0]
totalprizeamount = pdg.generate_var_prizeamount()

standard_loader_message = dmc.Loader(html.Div("Initializing tab..."))

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
    # {"value": 1825, "label": "1825"},
    # {"value": 1850, "label": "1850"},
    # {"value": 1875, "label": "1875"},
    {"value": 1900, "label": "1900"},
    # {"value": 1925, "label": "1925"},
    # {"value": 1950, "label": "1950"},
    # {"value": 1975, "label": "1975"},
    # {"value": 2000, "label": "2000"},
    {"value": int(lastyearincluded), "label": lastyearincluded}
]

##################################################################################################
# Dashboard Main Setup
##################################################################################################

# _dash_renderer._set_react_version("18.2.0")

app = Dash(
    external_stylesheets=[
        "assets/dmc_styles.css",
        dmc.styles.ALL
    ],
    title="Nobel Laureate Data Dashboard",
    suppress_callback_exceptions=True
)

server = app.server

##################################################################################################
# Plot Configuration Classes
##################################################################################################

class PlotConfig:
    def __init__(
        self,
        plot_id,
        header=None,
        subheader=None,
        data_range=None,
        show_filters=None,
        badges=None,
        chips_notchecked=None,
        chips_disabled =None,
        timerange=None,
        timerange_field=None,
        plot_generator=None,
        plot_generator_kwargs=None,
        footer=None,
        style=None,
    ):
        self.plot_id = plot_id
        self.header = header or "Generic Plot Title"
        self.subheader = subheader
        self.data_range = data_range or ("1901", lastyearincluded)
        self.show_filters = show_filters or {"categories": True, "gender": True, "timerange": True, "custom-filter": None}
        self.badges = badges or ["All Categories", f"1901-{lastyearincluded}"]
        self.chips_notchecked = chips_notchecked or []
        self.chips_disabled = chips_disabled or []
        self.timerange = timerange or ["1901", lastyearincluded]
        self.timerange_field = timerange_field or "award"
        self.plot_generator = plot_generator
        self.plot_generator_kwargs = plot_generator_kwargs or {}
        self.footer = footer or []
        self.style = style or {'width': '100%', 'height': '100%'}

    def get_plot_generator(self):
        if not self.plot_generator:
            raise ValueError("Plot generator function not defined.")
        return getattr(pdg, self.plot_generator, None)

    def generate_layout(self):
        return generate_plot_in_layout_class(self)
    
    def generate_badges(self):
        badges_code = []
        for badge in self.badges:
            badges_code.append(dmc.Badge(badge, variant="outline", color=pdg.c_brand_color_alt, mr="xs"))
        return badges_code

##################################################################################################
# Individual Plot Parameters
##################################################################################################

plot_configs = {
    # Choropleth Globe
    "fig_choroplethglobe_prizespercountry": PlotConfig(
        plot_id="fig_choroplethglobe_prizespercountry",
        header="Nobel Prizes by Country of Birth",
        subheader="This plot shows the distribution of country of birth of the laureates; you may rotate the globe and zoom in and out.",
        plot_generator="generate_choroplethglobe",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: The US dominance is clearly visible; apart from Africa, there are surprisingly few white spots.")
        ],
    ),

    # Scatter Mapbox Cities
    "fig_map_cities": PlotConfig(
        plot_id="fig_map_cities",
        header="Places of Birth and Death",
        subheader="This map shows the cities of birth and death of the laureates.",
        plot_generator="generate_scattermapbox_cities",
        plot_generator_kwargs={"data": df_laureates},
        show_filters={
            "categories": True,
            "gender": True,
            "custom-filter": dmc.Stack(
                children=[
                    html.Div("Type of city:"),
                    dcc.Dropdown(
                        id={
                            "type": "custom-filter",
                            "index": "fig_map_cities",
                            "filter": "city"
                        },
                        options=[
                            {'label': 'Birth', 'value': 'birth'},
                            {'label': 'Affiliation at Award', 'value': 'affiliation'},
                            {'label': 'Death', 'value': 'death'}
                        ],
                        value='birth',
                        clearable=False,
                        style={"width": "200px"}
                    ),
                ],
                gap="xs",
                align="flex-start",
                style={"margin-top": "0px"}
            )
        },
        footer=[
            dcc.Markdown("**Interesting Findings**: The map reveals significant global patterns in birth and death locations.")
        ],
    ),

    # Bubbles Per Population
    "fig_bubbles_population": PlotConfig(
        plot_id="fig_bubbles_population",
        header="Nobel Prizes by Country of Birth and Population",
        subheader="This plot shows the number of prizes by country of birth, but in relation to the population size of the country in the respective year.",
        plot_generator="generate_bubbles_perpopulation",
        plot_generator_kwargs={"data": df_prizes},
        footer=[
            dcc.Markdown("**Interesting Findings**: The visualization shows that countries like the USA only start to play an important role after World War II.")
        ],
    ),

    # Bar Per Country
    "fig_bar_prizespercountry": PlotConfig(
        plot_id="fig_bar_prizespercountry",
        header="Nobel Prizes by Country of Birth per Year",
        subheader="This plot shows the number of prizes per country of birth of laureates by year. You may deselect and reselect countries from the legend to customize your plot.",
        plot_generator="generate_bar_percountry",
        plot_generator_kwargs={"data": df_prizes},
        footer=[
            dcc.Markdown("**Interesting Findings**: The many pink lines on top in the right part of the plot emphasize our earlier finding that the number of prizes given to the USA has increased tremendously only after World War II.")
        ],
    ),

    "fig_bar_prizespercountry_rs": PlotConfig(
        plot_id="fig_bar_prizespercountry_rs",
        header="Nobel Prizes by Country of Birth per Year (Running Sum)",
        subheader="This plot shows the running sum of prizes by country of birth per year.",
        plot_generator="generate_bar_percountry",
        plot_generator_kwargs={"data": df_prizes, "runningsum": True},
        footer=[
            dcc.Markdown("**Interesting Findings**: This view highlights the cumulative advantage enjoyed by some countries over time.")
        ],
    ),

    # 3D Surface Per Gender
    "fig_surface_prizesforwomen": PlotConfig(
        plot_id="fig_surface_prizesforwomen",
        header = "Nobel Prizes Awarded to Women",
        subheader = "This plot shows the number of prizes for women in all of the disciplines per decade. It allows you to see when and in which disciplines the most prizes were awarded to women. Feel free to rotate the plot and zoom.",
        plot_generator="generate_3dsurface_pergender",
        plot_generator_kwargs={"data": df_laureates, "gender": "female"},
        show_filters={"categories": True, "gender": False, "custom-filter": False},
        footer=[
            dcc.Markdown("**Interesting Findings**: Although it looks like an impressively high mountain range in the more recent decades, note that the maximum value on the y-axis is 5, meaning that at the most 5 out of 30 possible prizes per decade/category went to women.")
        ],
    ),

    "fig_surface_prizesformenwomen": PlotConfig(
        plot_id="fig_surface_prizesformenwomen",
        header = "Nobel Prizes Awarded to Men and Women",
        subheader = "This plot is identical to the above, but here, men and women are both shown as two surfaces.",
        plot_generator="generate_3dsurface_pergender",
        plot_generator_kwargs={"data": df_laureates, "gender": "all"},
        show_filters={"categories": True, "gender": False, "custom-filter": False},
        footer=[
            dcc.Markdown("**Interesting Findings**: It is not much of a surprise that much many prizes were given to men than to women. Disciplines that perform particularly poorly are economics sciences and physics. The plot also clearly shows that the number of laureates (per year/decade) increases, i.e. prizes are more often given to two or three laureates instead of just one.")
        ],
    ),

    # Donut Charts
    "fig_donut_gender": PlotConfig(
        plot_id="fig_donut_gender",
        header="Gender Distribution",
        subheader = "The labels female and male are taken directly from the official Nobel Prize Outreach API. It would be interesting to learn how they get/set those values, or if they are simply based on perception. In any case, the cases where perception differs from self-identification may exist, but they will not substantially change the findings.",
        plot_generator="generate_donut",
        plot_generator_kwargs={"data": df_laureates, "characteristic": "gender"},
        show_filters={"categories": True, "gender": False, "custom-filter": False},
        footer = [
            dcc.Markdown("**Interesting Findings**: This plain old pie chart reaffirms what we found already above.")
            ],
    ),

    "fig_donut_ethnicity": PlotConfig(
        plot_id="fig_donut_ethnicity",
        header="Ethnicity Distribution",
        subheader = "This plot shows the distribution of ethnicities among Nobel Laureates. I am aware that notions of ethnicity or even race can be considered problematic. There are some who suggest to not use these categorizations at all. However, I think we may loose analytical power if we do; this chart is the successor to an earlier one that showed that there are exactly zero Black Nobel laureates in the natural sciences. This certainly is an interesing finding, how ever one may interpret it.",
        plot_generator="generate_donut",
        plot_generator_kwargs={"data": df_laureates, "characteristic": "ethnicity"},
        footer = [
            dcc.Markdown("**Assignment Process**: For additional transparency, here is how I have assigned the labels. Feel free to constructively critizice it. First, I started by geography: Everyone born in Europe was assigned *European*. As a starting point, everyone born in the USA or Canada was also assigned *European*. Similarly for all other continents. That process so far already raises difficult questions as to what ethnicity is, exactly. There is a myriad of publications on this topic, so my working definition was: Where someone's family originated from, going back to before Columbus. That then introduces two new categories for North America: *African-American*, and *North American*. Why not native American? Because even the native people of almost every country immigrated at some point in human history, as far as we know. Consequently, we then have *South American*, and then again *European* for all the (mostly) Spanish and Portuguese immigrants to South America. You may miss some categories like Central America, American Indians, Alaska Natives, etc - but there are simply no laureates in these ethnicities yet, so no need for further distinction. Israel is a special case: geographically, one would have to attribute *Asian*, but historically, most Israeli (laureates) have migrated there from parts of Europe. This is also an example for the next step (after categorization by continent), where I checked various lists available on the internet (mostly Wikipedia), such as \"List of Black Nobel Laureates\", \"List of Latin American Nobel Laureates\", and so on. Whenever appropriate, I changed the label. Next, I went through all the names one by one. Due to my former occupation, I know about 60 percent of them and also know the basics of their biographies. For the remaining ones, I checked their Wikipedia pages. You may note that there is also the category *Various*, which is a more subtle version of \"Mixed\". If it said, for example, on a laureate's Wikipedia page, that he had a British father and Korean mother, then I assigned *Various*.")
            ],
    ),

    "fig_donut_religion": PlotConfig(
        plot_id="fig_donut_religion",
        header="Religion Distribution",
        subheader = "This plot shows the distribution of religion among Nobel Laureates.",
        plot_generator="generate_donut",
        plot_generator_kwargs={"data": df_laureates, "characteristic": "religion"},
        footer = [
            html.Div(dcc.Markdown(["**Note**: Yet another sightly problematic categorization, for various reasons. One of them is data availability. There are lists on Wikipedia for Jewish, Muslim and Christian laureates, which I used. My suspicion here is though that the list of Jewish laureates is more or less complete, while that of Muslim laureates is not. For the list of Christian laureates, it states that it only lists laureates that have professed their faith. So this graph is actually somewhat misleading: First of all, it is unclear wether it is about \"firm faith\" or just religious upbringing. Second, we know little about what laureates really believe, which may be different from their religion. In any case: If we were to look at religion as stated in some official documents, then I suppose the number for Muslims should be higher, the number for Christians should be much higher (close to all of European Ethnicity), and we also have to add those religions completely lacking at the moment, e.g. Asian religions (and others)."])),
            html.Div(dcc.Markdown(["**Interesting Findings**: Even with the necessary changes described above, there still is a large number of Jewish Nobel laureates: 17.6 percent. According to Wikipedia, the Jewish religion has share among all religions in the world of 0.2 percent."]))   
        ],
    ),

    # Histogram Time Gap
    "fig_histogram_timegap": PlotConfig(
        plot_id="fig_histogram_timegap",
        header = "Timegap Between Discovery and Prize (Histogram)",
        subheader = "This histogram shows how often a value appears. For example, a waiting time of 11 years happened most often (=highest bar)",
        badges=["Natural Sciences", "1994-2014/2024"],
        chips_notchecked=["Economics", "Literature", "Peace"],
        chips_disabled=["Economics", "Literature", "Peace"],
        plot_generator_kwargs={"data":df_prizes, "categories":"natsci"},
        plot_generator = "generate_histogram_timegap",
        show_filters={
            "categories": True, 
            "gender": True,
            "custom-filter": dmc.Stack(
                    children=[
                        html.Div("Data source:"),
                        dcc.Dropdown(
                            id={
                                "type": "custom-filter", # custom filters must have 'custom-filter'
                                "index": "fig_histogram_timegap",  # must match the plot_id
                                "filter": "datasource"  # must match a plot generator function kwarg so it will be passed properly
                            },
                            options=[
                                {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                {'label': '1901 - 2023 (both)', 'value': 'both'}
                            ],
                            value='both',  # Default value
                            clearable=False,
                            style={"width": "200px"}
                        ),
                    ],
                gap="xs",  # space between the label and the dropdown
                align="flex-start",  # Align items to the left
                style={"margin-top": "0px"}
            )
        },
        footer = [
            dcc.Markdown("**Interesting Findings**: Most scientists who got the Nobel prize had to wait between 1 and 30 years, with a peak around 11 years. Very long waiting times don't appear very often.")
        ]
    ),

    # Scatterbox Time Gap Trend
    "fig_scatter_timegap_trend": PlotConfig(
        plot_id = "fig_scatter_timegap_trend",
        header = "Timegap Between Discovery and Prize (Trendlines)",
        subheader = "This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up).",
        badges=["Natural Sciences", f"1901-{lastyearincluded}"],
        chips_notchecked=["Economics", "Literature", "Peace"],
        chips_disabled=["Economics", "Literature", "Peace"],
        plot_generator_kwargs={"data":df_prizes, "categories":"natsci"},
        plot_generator = "generate_scatterbox_timegaptrend",
        show_filters={
            "categories": True, 
            "gender": True,
            "custom-filter": dmc.Stack(
                    children=[
                        html.Div("Data source:"),
                        dcc.Dropdown(
                            id={
                                "type": "custom-filter", # custom filters must have 'custom-filter'
                                "index": "fig_scatter_timegap_trend",  # must match the plot_id
                                "filter": "datasource"  # must match a plot generator function kwarg so it will be passed properly
                            },
                            options=[
                                {'label': '1901 - 2014 (Nature paper)', 'value': 'paper'},
                                {'label': '2015 - 2023 (ChatGPT)', 'value': 'chatgpt'},
                                {'label': '1901 - 2023 (both)', 'value': 'both'}
                            ],
                            value='both',  # Default value
                            clearable=False,
                            style={"width": "200px"}
                        ),
                    ],
                gap="xs",  # space between the label and the dropdown
                align="flex-start",  # Align items to the left
                style={"margin-top": "0px"}
            )
        },
        footer = [
            dcc.Markdown("**Interesting Findings**: It seems that the time gap increases in a pretty similar fashion as the life expectancy.")
            ],
    ),


    # Scatterbox Age
    "fig_scatterbox_age": PlotConfig(
        plot_id="fig_scatterbox_age",
        header = "Laureate Age at Time of Award",
        subheader = "This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up).",
        plot_generator="generate_scatterbox_age",
        plot_generator_kwargs={"data": df_laureates},
        footer = [dcc.Markdown("**Interesting Findings**: In the early years, the average age in the natural sciences was around 45, whereas nowadays it is close to 65. This fits well to the earlier finding that the timegap has increased by - on average - 25 years. Interestingly enough, peace prize awardees get younger.")],
    ),

    # Heatmap Age
    "fig_heatmap_age": PlotConfig(
        plot_id="fig_heatmap_age",
        header = "Laureate Age at Time of Award (Heatmap)",
        subheader = "Same data as above, but displayed as heatmap.",
        plot_generator="generate_heatmap_age",
        plot_generator_kwargs={"data": df_laureates},
        footer=[],
    ),

    # Parcat Migration
    "fig_parcat_migration_dwp": PlotConfig(
        plot_id="fig_parcat_migration_dwp",
        header = "Movement: Place of Main Degree / Main Discovery / Prize",
        subheader = [
            dcc.Markdown("This plot shows the movement between three locations: where did the laureates get their main university degree (or similar), where did they do their main work that led to the discovery, and where did they work at the time when they received the prize? This plot is based on the Nature paper \"At what institutions did Nobel laureates do their prize-winning work?\" (see References), which unfortunately only covers the years 1994 - 2014."),
            dcc.Markdown("**How to Read:**: The three vertical pillars stand for the three points and places in time: **degree, work, prize**. The lines show the flow from one place to the next. The on-hover infobox also shows you the overall percentage of the selected group. If you like, you may also re-arrange the bar sections via drag and drop. The dropdowns let you choose between *City* (many), *Country* (less), and the combination of both, which distinguishes Cambridge UK from Cambridge USA (etc.)")
        ],
        badges=["Natural Sciences", f"1901-{lastyearincluded}"],
        chips_notchecked = ["Economics", "Literature", "Peace"],
        chips_disabled = ["Economics", "Literature", "Peace"],
        plot_generator="generate_parcat_migration",
        show_filters={
            "categories": True, 
            "gender": True,
            "custom-filter": dmc.Stack(
                    children=[
                        html.Div("Type of city:"),
                        dmc.Group(
                            children=[
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_dwp",  # must match the plot_id
                                        "filter": "loc1"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'ParCatDegreeCity'},
                                        {'label': 'Country', 'value': 'ParCatDegreeCountry'},
                                        {'label': 'City+Country', 'value': 'ParCatDegreeCityCountry'}
                                    ],
                                    value='ParCatDegreeCountry',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_dwp",  # must match the plot_id
                                        "filter": "loc2"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'ParCatWorkCity'},
                                        {'label': 'Country', 'value': 'ParCatWorkCountry'},
                                        {'label': 'City+Country', 'value': 'ParCatWorkCityCountry'}
                                    ],
                                    value='ParCatWorkCountry',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_dwp",  # must match the plot_id
                                        "filter": "loc3"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'ParCatPrizeCity'},
                                        {'label': 'Country', 'value': 'ParCatPrizeCountry'},
                                        {'label': 'City+Country', 'value': 'ParCatPrizeCityCountry'}
                                    ],
                                    value='ParCatPrizeCountry',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                            ]
                        )
                    ],
                gap="xs",  # space between the label and the dropdown
                align="flex-start",  # Align items to the left
                style={"margin-top": "0px"}
            )
        },
        footer = [
            dcc.Markdown("**Interesting Findings**: There are many findings to be made: For example, US laureates tend to be very immobile; however, not as immobile as the French. German researchers, on the other hand, love to go abroad - however you may want to interpret that. Finally, it is an interesting exercise to speculate if the period 1994-2014 is significantly different from other periods.")
        ],
    ),
    "fig_parcat_migration_bpd": PlotConfig(
        plot_id="fig_parcat_migration_bpd",
        header = "Movement: Birth / Prize / Death",
        subheader = "This plot works the same way, but has slightly diffferent data: place of birth, place of organisation when the prize was awarded, place of death. Note that this dataset, unlike the previous one, spans the full time range. (Selecting *City* may lead to incorrect visuals, as there are simply too many to display.)",
        badges=["Natural Sciences", f"1901-{lastyearincluded}"],
        plot_generator="generate_parcat_migration",
        plot_generator_kwargs={"data": df_laureates},
        show_filters={
            "categories": True, 
            "gender": True,
            "custom-filter": dmc.Stack(
                    children=[
                        html.Div("Type of city:"),
                        dmc.Group(
                            children=[
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_bpd",  # must match the plot_id
                                        "filter": "loc1"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'BirthCityNow'},
                                        {'label': 'Country', 'value': 'BirthCountryNow'},
                                        {'label': 'Continent', 'value': 'BirthContinent'},
                                    ],
                                    value='BirthContinent',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_bpd",  # must match the plot_id
                                        "filter": "loc2"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'Prize0_Affiliation0_CityNow'},
                                        {'label': 'Country', 'value': 'Prize0_Affiliation0_Country'},
                                        {'label': 'Continent', 'value': 'Prize0_Affiliation0_Continent'}
                                    ],
                                    value='Prize0_Affiliation0_Country',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                                dcc.Dropdown(
                                    id={
                                        "type": "custom-filter", # custom filters must have 'custom-filter'
                                        "index": "fig_parcat_migration_bpd",  # must match the plot_id
                                        "filter": "loc3"  # must match a plot generator function kwarg so it will be passed properly
                                    },
                                    options=[
                                        {'label': 'City', 'value': 'DeathCityNow'},
                                        {'label': 'Country', 'value': 'DeathCountryNow'},
                                        {'label': 'Continent', 'value': 'DeathContinent'}
                                    ],
                                    value='DeathCityNow',  # Default value
                                    clearable=False,
                                    style={"width": "200px"}
                                ),
                            ]
                        )
                    ],
                gap="xs",  # space between the label and the dropdown
                align="flex-start",  # Align items to the left
                style={"margin-top": "0px"}
            )
        },
        footer = [
            dcc.Markdown("**Interesting Findings**: There are many findings to be made: For example, US laureates tend to be very immobile; however, not as immobile as the French. German researchers, on the other hand, love to go abroad - however you may want to interpret that. Finally, it is an interesting exercise to speculate if the period 1994-2014 is significantly different from other periods.")
        ],
    ),

     # Line Prize Money
    "fig_line_prizemoney": PlotConfig(
        plot_id="fig_line_prizemoney",
        header="Prize Money Over the Years",
        subheader = f"In SEK; total amount paid up until today: {(totalprizeamount * 0.088):,.0f} EUR",
        plot_generator="generate_line_prizemoney",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: Prize money has increased significantly in recent decades.")
        ],
    ),

    # Sunburst
    "fig_sunburst_last": PlotConfig(
        plot_id="fig_sunburst_last",
        header="Discipline - Gender - Country",
        subheader="Click on the segments to filter the data.",
        badges=["All Categories", f"{lastyearincluded}"],
        plot_generator="generate_sunburst",
        plot_generator_kwargs={"data": df_laureates, "year": "last"},
        footer=[
            dcc.Markdown("**Interesting Findings**: The 2024 prizes were mostly given to male researchers from the US or UK.")
        ],
    ),

    # Globe Movement
    "fig_globe_movement": PlotConfig(
        plot_id="fig_globe_movement",
        header = "Movement: Birth / Prize",
        subheader = "This globe shows the movement from place of birth to place of affiliation at the time of the award.)",
        plot_generator="generate_globe_movement",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: Isn't it nice to look at?")
        ],
    ),

    # Map Movement
    "fig_map_movement": PlotConfig(
        plot_id="fig_map_movement",
        header="Life Paths (Birth - Work)",
        subheader="Some laureates haven't moved and are represented as dots.",
        badges=["All Categories", f"{lastyearincluded}"],
        plot_generator="generate_map_movement",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: Common patterns: European researchers move to the US, the Americans switch coasts at most, and the Asians stay where they are. No South Americans or Africans.")
        ],
    ),

    # Most Common Firstnames
    "fig_mostcommon_firstnames": PlotConfig(
        plot_id="fig_mostcommon_firstnames",
        header="Most Common Firstnames",
        subheader="What to name your kid if you want it to become a Nobel laureate.",
        plot_generator="generate_mostcommon_firstnames",
        #plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: We may assume that the name itself will not have much of an influence on the laureates' success. However, the names are a good indicator of the social and economic status, as well as nationality and of course gender.")
        ],
    ),

    

}


##################################################################################################
# Functions to generate plots in the layout
##################################################################################################

# All functions get their parameters from the PlotConfig instance provided to them (as id). A list is provided further up.
# If you want to change the plot parameters, you need to change the plot_configs dictionary.

#1: Create loading-spinners (placeholders) for the plots while they are being generated
def generate_loader_spinner(id):
    return html.Div(
        id={'type':'outer-container', 'index':id},
        children=[
            html.Div(
                dmc.Loader(
                    id={'type':'spinner', 'index':id},
                    color= pdg.c_grey,
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

# Callback to display the plots once they are generated
@app.callback(
    Output({'type': 'inner-container', 'index': MATCH}, "children"),
    Output({'type': 'spinner', 'index': MATCH}, "style"),  # Hide loader
    Output({'type': 'border', 'index': MATCH}, "style"),  # Hide border
    Output({'type': 'inner-container', 'index': MATCH}, "style"),  # Show graph
    Input({'type': 'inner-container', 'index': MATCH}, "id"),  # Trigger on app load
)
def display_plot(triggered_id):
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


#2: Create the layout for the plots. Once the plots are generated, the spinner will be replaced by the actual plot.
def generate_plot_in_layout_class(plot_config):
    """
    Generates the layout for a plot using the provided PlotConfig instance.
    """

    # Resolve the plot generator function, i.e. the function that generates the plot, stored in the class instance (dict)
    plot_generator = plot_config.get_plot_generator()
    if not plot_generator:
        raise ValueError(f"Plot generator '{plot_config.plot_generator}' not found in module.")

    # Generate the initial figure using the resolved generator and standard kwargs, stored in the class instance (dict)
    figure = plot_generator(**plot_config.plot_generator_kwargs)



    # Return the layout for the plot
    return dmc.SimpleGrid(
        cols={"base": 1, "sm": 1},
        spacing="sm",
        verticalSpacing="sm",
        children=[
            html.Div(
                id=plot_config.plot_id,
                children =[
                    # Header
                    html.Div(
                        [
                            html.H3(plot_config.header, className="plot-header"),

                            dmc.Grid(
                                columns=24,
                                children=[
                                    dmc.GridCol(html.Div(plot_config.subheader, className="plot-subheader"), span={"base": 24, "md": 14}),
                                    dmc.GridCol(html.Div(plot_config.generate_badges()), span={"base": 24, "md": 7}),
                                    dmc.GridCol(
                                        
                                        # Filter Modal
                                        html.Div(
                                            [
                                                dmc.Button(
                                                    "Filter", 
                                                    variant="gradient", 
                                                    gradient={"from": pdg.c_blue_light, "to": pdg.c_green_dark}, 
                                                    size="xs", 
                                                    id={"type": "filter-button", "index": plot_config.plot_id},
                                                    className="filter-button"
                                                ),
                                                dmc.Drawer(
                                                    title="Filter",
                                                    position="right",
                                                    id={"type": "filter-modal", "index": plot_config.plot_id},
                                                    size="300px",
                                                    style={"display": "block"},
                                                    children=[
                                                        # Selection Area
                                                        dmc.Stack(
                                                            [
                                                                # Category filter
                                                                dmc.Stack(
                                                                    children=[
                                                                        html.Div("Prize Categories"),
                                                                        html.Div(
                                                                            dmc.Group(
                                                                                [
                                                                                    dmc.Chip("Medicine", size="xs", variant="outline", checked=False if "Medicine" in plot_config.chips_notchecked else True, disabled=True if "Medicine" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-medicine", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("Physics", size="xs", variant="outline", checked=False if "Physics" in plot_config.chips_notchecked else True, disabled=True if "Physics" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-physics", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("Chemistry", size="xs", variant="outline", checked=False if "Chemistry" in plot_config.chips_notchecked else True, disabled=True if "Chemistry" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-chemistry", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("Economics", size="xs", variant="outline", checked=False if "Economics" in plot_config.chips_notchecked else True, disabled=True if "Economics" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-economics", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("Literature", size="xs", variant="outline", checked=False if "Literature" in plot_config.chips_notchecked else True, disabled=True if "Literature" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-literature", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("Peace", size="xs", variant="outline", checked=False if "Peace" in plot_config.chips_notchecked else True, disabled=True if "Peace" in plot_config.chips_disabled else False, color=pdg.c_brand_color_main, id={"type": "chip-peace", "index": plot_config.plot_id}),
                                                                                ]
                                                                            )
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "marginRight": "30px",
                                                                        "display": "block" if plot_config.show_filters.get("categories", False) else "none"
                                                                    }
                                                                ),
                                                                # Gender Filter
                                                                dmc.Stack(
                                                                    children=[
                                                                        html.Div("Gender"),
                                                                        html.Div(
                                                                            dmc.Group(
                                                                                [
                                                                                    dmc.Chip("female", size="xs", variant="outline", checked=True, color=pdg.c_brand_color_main, id={"type": "chip-female", "index": plot_config.plot_id}),
                                                                                    dmc.Chip("male", size="xs", variant="outline", checked=True, color=pdg.c_brand_color_main, id={"type": "chip-male", "index": plot_config.plot_id}),
                                                                                ]
                                                                            )
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
                                                                            dmc.Group(
                                                                                [
                                                                                    dcc.Dropdown(
                                                                                        id={"type": "dropdown-timerange", "index": plot_config.plot_id},
                                                                                        options=[
                                                                                            {'label': 'Year of Birth', 'value': 'birth'},
                                                                                            {'label': 'Year of Award', 'value': 'award'},
                                                                                            {'label': 'Year of Death', 'value': 'death'}
                                                                                        ],
                                                                                        value='award',  # Default value
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
                                                                                        color=pdg.c_teal
                                                                                        
                                                                                    ),
                                                                                ],
                                                                                mb=35
                                                                            )
                                                                        )
                                                                    ],
                                                                    #justify="left",
                                                                    style={"margin-top": "0px", "width":"100%"}
                                                                ),



                                                                # Custom Filter (if provided)
                                                                html.Div(
                                                                    plot_config.show_filters.get("custom-filter"),
                                                                    style={
                                                                        "display": "block" if plot_config.show_filters.get("custom-filter") else "none"
                                                                    }
                                                                ),

                                                                dmc.Space(h="lg"),

                                                                dmc.Switch(
                                                                    id={"type":"switch-update", "index": plot_config.plot_id},
                                                                    size="sm",
                                                                    radius="xl",
                                                                    label="Update plot only on Submit",
                                                                    checked=False
                                                                )
                                                            ],
                                                            gap="sm",
                                                            className="selection-area",
                                                        ),
                                                        # Filter modal buttons
                                                        dmc.Group(
                                                            [
                                                                dmc.Button("Submit", id={"type": "submit-button", "index": plot_config.plot_id}, color=pdg.c_brand_color_main),
                                                                dmc.Button(
                                                                    "Close",
                                                                    color="c_red",
                                                                    variant="outline",
                                                                    id={"type": "close-button", "index": plot_config.plot_id},
                                                                ),
                                                            ],
                                                            justify="flex-end",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        className="filter-button"
                                        ),
                                        span={"base": 24, "md": 3},
                                        # style={"alignItems": "center", "display": "flex", "justifyContent": "flex-end", "verticalAlign": "top"}
                                    ),
                                ]
                            )


                           # html.P(plot_config.subheader) if plot_config.subheader else None,
                        ]
                    ),




                    # Plot
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                id={"type": "plot", "index": plot_config.plot_id},
                                figure=figure,
                                style=plot_config.style,
                            )
                        ),
                        className="widget-content",
                    ),

                    # Footer
                    html.Div(
                        plot_config.footer,
                        className="widget-footer",
                    ) if plot_config.footer else None,
                ],
                className="widget-container",
            ),
        ],
    )

# Callback to update the marks of the timerange slider based on the selected value
# @app.callback(
#     Output({"type": "slider-timerange", "index": MATCH}, "marks"),
#     Input({"type": "dropdown-timerange", "index": MATCH}, "value")
# )
# def update_marks(selected_value):
#     if selected_value == "award":
#         return marks_award
#     else:
#         return marks_life

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
        Input({"type": "slider-timerange", "index": MATCH}, "value"),
        Input({"type": "dropdown-timerange", "index": MATCH}, "value"),
        Input({"type": "switch-update", "index": MATCH}, "checked"),
        Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
        Input({"type": "custom-filter", "index": MATCH, "filter": ALL}, "value")
    ],
    [State({"type": "plot", "index": MATCH}, "id")]
)
def update_plot(
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    chip_female, chip_male, slider_timerange, dropdown_timerange, switch_update, n_clicks, custom_filter_values, plot_id
):
    # Ensure the callback is only triggered when the submit button is clicked
    ctx = dash.callback_context

    if switch_update:
        if not any('"type":"submit-button"' in trigger['prop_id'] for trigger in ctx.triggered):
            raise PreventUpdate

    # Extract the plot configuration
    plot_id = plot_id["index"]
    plot_config = plot_configs.get(plot_id)
    if not plot_config:
        raise ValueError(f"Plot configuration for ID {plot_id} not found.")

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

    # print("Updater: ctx.inputs.keys():", ctx.inputs.keys())

    # # Add any custom filter values if they exist
    # if custom_filter_values:
    #     # Get all input IDs
    #     input_ids = [
    #         key for key in ctx.inputs.keys() 
    #         if isinstance(json.loads(key.split('.')[0]), dict) and 
    #         json.loads(key.split('.')[0]).get("type") == "custom-filter"
    #     ]
        
    #     # Parse the pattern IDs to get filter names
    #     custom_filter_patterns = [json.loads(input_id.split('.')[0]) for input_id in input_ids]
        
    if custom_filter_values:
        # Parse JSON once and filter for custom-filter inputs
        custom_filter_patterns = [
            json.loads(key.split('.')[0]) for key in ctx.inputs.keys()
            if isinstance(json.loads(key.split('.')[0]), dict) and 
            json.loads(key.split('.')[0]).get("type") == "custom-filter"
        ]


        # Add custom filter values to kwargs using the filter name from the pattern
        for pattern, value in zip(custom_filter_patterns, custom_filter_values):
            if value is not None:  # Only add non-None values
                filter_name = pattern["filter"]  # This gets "city" from the pattern
                plot_generator_kwargs[filter_name] = value

    # print("Updater: kwargs:", plot_generator_kwargs)

    # Generate the updated figure
    try:
        plot_generator = plot_config.get_plot_generator()
        updated_figure = plot_generator(**plot_generator_kwargs)
    except Exception as e:
        print(f"Error generating plot for {plot_id}: {e}")
        updated_figure = {"data": [], "layout": {"title": "Error generating plot"}}

    return updated_figure


# function to include pngs in the layout - should not be used, as the dashboard should only include plotly figures
def generate_png_in_layout(
    cols= {"base": 1, "sm": 1},
    header= "Generic Plot Title", 
    subheader="", 
    datafrom="1901", 
    datato=lastyearincluded, 
    badges=[dmc.Badge("All Categories", variant="outline", color= pdg.c_brand_color_alt)],
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
                                    dmc.Badge(f"{datafrom} - {datato}", variant="outline", color= pdg.c_brand_color_alt),
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

def render_overview_content():
    return dmc.Stack(
        id="section-overview",
        children=[
            dmc.Stack(
                children = [
                    html.Div("Select Categories"),
                    html.Div(
                        dmc.Group(
                            [
                                dmc.Chip("Medicine", checked=True, color=pdg.c_medicine, id="chip-medicine"),
                                dmc.Chip("Physics", checked=True, color=pdg.c_physics, id="chip-physics"),
                                dmc.Chip("Chemistry", checked=True, color=pdg.c_chemistry, id="chip-chemistry"),
                                dmc.Chip("Economics", checked=True, color=pdg.c_economics, id="chip-economics"),
                                dmc.Chip("Literature", checked=True, color=pdg.c_literature, id="chip-literature"),
                                dmc.Chip("Peace", checked=True, color=pdg.c_peace, id="chip-peace")
                            ]
                        )

                    ),

                    html.Div("Select Gender"),
                    dmc.Stack(
                        children=[
                            html.Div(
                                dmc.Group(
                                    [
                                        dmc.Chip("female", variant="outline", checked=True, color=pdg.c_red, id="chip-female"),
                                        dmc.Chip("male", variant="outline", checked=True, color=pdg.c_teal, id="chip-male"),
                                    ]
                                )
                            )
                        ]
                    ),

                    html.Div("Select Time Range"),
                    html.Div(
                        [
                            dmc.RangeSlider(
                                id="slider-timerange-overview",
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
                                color=pdg.c_teal
                            ),

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
            )
        ]
    )





# Dynamic Content for the Overview Tab based of the selected filter values
# ##################################################################################################

@app.callback(
    Output("overview-content", "children"), 
    [   Input("chip-medicine", "checked"),
        Input("chip-physics", "checked"),
        Input("chip-chemistry", "checked"),
        Input("chip-economics", "checked"),
        Input("chip-literature", "checked"),
        Input("chip-peace", "checked"),
        Input("chip-female", "checked"),
        Input("chip-male", "checked"),
        Input("slider-timerange-overview", "value")],
        # prevent_initial_call=True
)
def update_overview_content(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, chip_female, chip_male, timerange):

    selected_categories = pdg.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    # print("Selected Categories:", selected_categories)
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
                            dcc.Graph(id="fig_donut_gender_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="gender"), style={'width': '100%', 'height':'100%'}),
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
                            dcc.Graph(id="fig_donut_ethnicity_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="ethnicity"), style={'width': '100%', 'height':'100%'}),
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
                            dcc.Graph(id="fig_donut_religion_overview", figure=pdg.generate_donut(data=df_filtered_laureates, characteristic="religion"), style={'width': '100%', 'height':'100%'}),
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
                            dcc.Graph(id="fig_sunburst_overview", figure=pdg.generate_sunburst(data=df_filtered_laureates), style={'width': '100%', 'height':'100%'}),
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
# Section Current
##################################################################################################

current_prizes = pdg.get_currentlaureatemotivations(df_prizes)

def render_current_content():
    return dmc.Stack(
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
                                ),
    
                                html.Div(
                                    dcc.Markdown(current_prizes[0], dangerously_allow_html=True),
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
                                ),
    
                                html.Div(
                                    dcc.Markdown(current_prizes[1], dangerously_allow_html=True),
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
                                    dcc.Markdown(current_prizes[2], dangerously_allow_html=True),
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
                                ),
    
                                html.Div(
                                    dcc.Markdown(current_prizes[3], dangerously_allow_html=True),
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
                                ),
    
                                html.Div(
                                    dcc.Markdown(current_prizes[4], dangerously_allow_html=True),
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
                                ),

                                html.Div(
                                    dcc.Markdown(current_prizes[5], dangerously_allow_html=True),
                                    className="widget-content"
                                ),
                            ],
                            className="widget-container",
                        ),

                    ]
                ),
                
                generate_loader_spinner("fig_sunburst_last"),

                generate_loader_spinner("fig_map_movement"),

            ],
            gap="sm"
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
                                dmc.Chip("Medicine", checked=True, color=pdg.c_medicine, id="chip-medicine"),
                                dmc.Chip("Physics", checked=True, color=pdg.c_physics, id="chip-physics"),
                                dmc.Chip("Chemistry", checked=True, color=pdg.c_chemistry, id="chip-chemistry"),
                                dmc.Chip("Economics", checked=True, color=pdg.c_economics, id="chip-economics"),
                                dmc.Chip("Literature", checked=True, color=pdg.c_literature, id="chip-literature"),
                                dmc.Chip("Peace", checked=True, color=pdg.c_peace, id="chip-peace")
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
                                                    dmc.Chip("female", variant="outline", checked=True, color=pdg.c_red, id="chip-female"),
                                                    dmc.Chip("male", variant="outline", checked=True, color=pdg.c_teal, id="chip-male"),
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
                                                    color=pdg.c_teal,
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
                                                color=pdg.c_teal
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
                                                color=pdg.c_teal
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
            "flex": 1            # Spalten füllen verfügbaren Platz
        },
        dashGridOptions={
            "pagination": True,
            "autoSizeStrategy": {
                "type": "fitGridWidth",  # Spalten füllen Grid-Breite
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
        "height": "60vh",  # 60% der Viewport-Höhe
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
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';', encoding="UTF-8")

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';', encoding="UTF-8")

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
            html.Div(dcc.Markdown(["Schlagberger, E.M., Bornmann, L. & Bauer, J.: \"At what institutions did Nobel laureates do their prize-winning work? An analysis of biographical information on Nobel laureates from 1994 to 2014\". Scientometrics 109, 723–767 (2016). [Link](https://doi.org/10.1007/s11192-016-2059-2)"]), className="text-copy"),

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

            dmc.Space(h="xl"),

            html.H5("Version History", className="text-subheader"),
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

def create_nav_link(label, value, url, icon=None):
    return dcc.Link(
        dmc.NavLink(
            label=label,
            leftSection=dmc.ThemeIcon(DashIconify(icon=icon, width=20), size="sm", variant="light", color=pdg.c_brand_color_main) if icon else None,
            style={"cursor": "pointer"},
            mb="xs"
        ),
        href=url,
        style={"textDecoration": "none", "color": "inherit"}
    )

nav_items = [
    {"label": "Overview", "value": "overview", "url": "/overview", "icon": "material-symbols:space-dashboard-outline"},
    {"label": f"{lastyearincluded} Prizes", "value": "current", "url": "/current", "icon": "material-symbols:trophy-outline-rounded"},
    {"label": "Geography", "value": "geography", "url": "/geography", "icon": "material-symbols:globe-location-pin-rounded"},
    {"label": "Demography", "value": "demography", "url": "/demography", "icon": "material-symbols:group-outline-rounded"},
    {"label": "Time Analysis", "value": "time", "url": "/time", "icon": "material-symbols:calendar-clock-outline-rounded"},
    {"label": "Migration", "value": "migration", "url": "/migration", "icon": "material-symbols:houseboat-outline-rounded"},
    {"label": "Miscellaneous", "value": "misc", "url": "/misc", "icon": "material-symbols:award-star-outline-rounded"},
    {"label": "List Generator", "value": "list_generator", "url": "/list", "icon": "material-symbols:blur-linear-outline-rounded"},
    {"label": "Data & References", "value": "data", "url": "/data", "icon": "material-symbols:data-table-outline-rounded"},
]

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

layout = dmc.AppShell(
    [
        dcc.Location(id="url", refresh=False),
        # html.Div(id="scroll-dummy", style={"display": "none"}),
        dmc.AppShellHeader(
            dmc.Group(
                [
                    dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                    html.Img(src="assets/logo-md.png", style={"width": "120px", "height": "40px"}),
                    dmc.Title("Nobel Laureate Data Dashboard", order=2, c=pdg.c_brand_color_main),
                    dmc.Space(style={"flex": 1}),
                    dmc.Badge("v1.8", variant="light", color="blue"),
                ],
                h="100%",
                px="md",
                justify="space-between",
                align="center"
            ),
            style={"borderBottom": f"1px solid {pdg.c_grey_light}"}
        ),
        dmc.AppShellNavbar(
            id="navbar",
            children=[
                dmc.Stack([
                    dmc.Title("Navigation", order=4, mb="md", c=pdg.c_brand_color_main),
                    *[create_nav_link(item["label"], item["value"], item["url"], item.get("icon")) for item in nav_items]
                ], gap="xs")
            ],
            p="md",
        ),
        dmc.AppShellMain(
            dmc.Container(
                html.Div(id="main-content", children=[
                    # dmc.Stack([
                    #     dmc.Title("Hej.", order=1),
                    #     dmc.Text("This dashboard provides a great variety of plots, analyses and tables " \
                    #     "related to the Nobel Prize. Its core data comes from the official API, " \
                    #     "guaranteeing maximum data quality. " \
                    #     "In addition, various additional data sources have been integrated. " \
                    #     "To explore the plots, choose your preferred category from the menu." \
                    #     "The latest feature addition is the list generator, which offers a variety of filtering options," \
                    #     "and also lets you download the results.", size="md", style={"maxWidth": "700px"},),

                    #     dmc.SimpleGrid(
                    #         cols={"base": 1, "sm": 2, "md": 3},
                    #         spacing="xl",
                    #         style={"maxWidth": "700px"},
                    #         p="md",
                    #         children=[
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_cubes_fields.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_parcat.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/jsglobe.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_globe_birth.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_sunburst.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_map_cities.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_surface_mw.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_bubbles.png", radius="md", style={"margin": "auto"}),
                    #             dmc.Image(h=200, w="auto", fit="contain", src="assets/images/fig_countries_year.png", radius="md", style={"margin": "auto"}),                           
                    #         ]
                    #     ),

                    #     dmc.Text(
                    #         children=[
                    #             "For more details, please refer to the ",
                    #             dcc.Link("Data & References", href="/data", style={"textDecoration": "underline"}),
                    #             " section."
                    #         ],
                    #         style={"maxWidth": "700px"}
                    # )

                    # ], align="left", mt="xl")
                ]),
                fluid=True,
                p="md"
            )
        ),
    ],
    header={"height": 100},
    navbar={
        "width": 220,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(layout)

##################################################################################################
# Callbacks
##################################################################################################

# Burger menu callback
@app.callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def navbar_is_open(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar

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


##################################################################################################
# Running the app
##################################################################################################

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    debug_mode = os.environ.get('DEBUG', 'True').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)