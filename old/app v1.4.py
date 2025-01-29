##################################################################################################
# Library Imports
##################################################################################################

import pandas as pd
import dash_ag_grid as dag
import os
import dash
from dash import dcc, html, Dash, _dash_renderer, State
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, MATCH, ALL
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



##################################################################################################
# Color Settings
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
# Load Data
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

# Nobel Prize Stats
df_prizestats = pd.read_csv("df_prizestats.csv", sep=';', encoding="UTF-8")

df=pdg.count_per_country()
max_prize_count = df['Count'].max()
lastyearincluded = 2024
numberofprizes = df_prizes.shape[0]

totalprizeamount = pdg.generate_var_prizeamount()

standard_loader_message = dmc.Loader(html.Div("Initializing tab..."))

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
# Plot Class
##################################################################################################

class PlotConfig:
    def __init__(
        self,
        plot_id,
        header="Generic Plot Title",
        subheader="",
        data_range=("1901", lastyearincluded),
        show_filters={"categories": True, "gender": True, "custom-filter": None},
        badges=None,
        plot_generator=None,
        plot_generator_kwargs=None,
        footer=None,
        style=None,
    ):
        self.plot_id = plot_id
        self.header = header
        self.subheader = subheader
        self.data_range = data_range
        self.show_filters = show_filters
        self.badges = badges or [dmc.Badge("All Categories", variant="outline", color=brand_color_alt)]
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
                            {'label': 'City of Birth', 'value': 'birth'},
                            {'label': 'City of Affiliation at Time of Award', 'value': 'affiliation'},
                            {'label': 'City of Death', 'value': 'death'}
                        ],
                        value='birth',
                        clearable=False,
                        style={"width": "400px"}
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
        badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
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
                            style={"width": "400px"}
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
        badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
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
                            style={"width": "400px"}
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
        badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
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
        badges = [dmc.Badge("Natural Sciences", variant="outline", color=brand_color_alt)],
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
        badges = [dmc.Badge("All Categories", variant="outline", color=brand_color_alt)],
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
        plot_generator="generate_map_movement",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: Common patterns: European researchers move to the US, the Americans switch coasts at most, and the Asians stay where they are. No South Americans or Africans.")
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
                    color= "#e6e6e6",
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
                [
                    # Header
                    html.Div(
                        [
                            html.H3(plot_config.header),
                            html.P(plot_config.subheader) if plot_config.subheader else None,
                        ]
                    ),

                    # Badges
                    dmc.Stack(
                        children=[
                            html.Div(plot_config.badges)
                        ],
                        style={
                            "marginBottom": "10px",
                        }
                    ),

                    # Filter Modal
                    html.Div(
                        [
                            dmc.Button(
                                "Filter", 
                                variant="gradient", 
                                gradient={"from": c_lightblue, "to": c_teal}, 
                                size="xs", 
                                id={"type": "filter-button", "index": plot_config.plot_id}
                            ),
                            dmc.Modal(
                                title="Filter",
                                centered=True,
                                id={"type": "filter-modal", "index": plot_config.plot_id},
                                size="750px",
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
                                                                dmc.Chip("Medicine", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-medicine", "index": plot_config.plot_id}),
                                                                dmc.Chip("Physics", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-physics", "index": plot_config.plot_id}),
                                                                dmc.Chip("Chemistry", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-chemistry", "index": plot_config.plot_id}),
                                                                dmc.Chip("Economics", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-economics", "index": plot_config.plot_id}),
                                                                dmc.Chip("Literature", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-literature", "index": plot_config.plot_id}),
                                                                dmc.Chip("Peace", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-peace", "index": plot_config.plot_id}),
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
                                                                dmc.Chip("female", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-female", "index": plot_config.plot_id}),
                                                                dmc.Chip("male", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-male", "index": plot_config.plot_id}),
                                                            ]
                                                        )
                                                    ),
                                                ],
                                                style={
                                                    "display": "block" if plot_config.show_filters.get("gender", False) else "none"
                                                }
                                            ),

                                            # Custom Filter (if provided)
                                            html.Div(
                                                plot_config.show_filters.get("custom-filter"),
                                                style={
                                                    "display": "block" if plot_config.show_filters.get("custom-filter") else "none"
                                                }
                                            ),
                                        ],
                                        gap="sm",
                                        className="selection-area",
                                    ),
                                    # Filter modal buttons
                                    dmc.Group(
                                        [
                                            dmc.Button("Submit", id={"type": "submit-button", "index": plot_config.plot_id}),
                                            dmc.Button(
                                                "Close",
                                                color="red",
                                                variant="outline",
                                                id={"type": "close-button", "index": plot_config.plot_id},
                                            ),
                                        ],
                                        justify="flex-end",
                                    ),
                                ],
                            ),
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
        Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
        Input({"type": "custom-filter", "index": MATCH, "filter": ALL}, "value")
    ],
    [State({"type": "plot", "index": MATCH}, "id")]
)
def update_plot(
    chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace,
    chip_female, chip_male, n_clicks, custom_filter_values, plot_id
):
    # Ensure the callback is only triggered when the submit button is clicked
    ctx = dash.callback_context
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
        "gender": selected_genders
    })

    print("Updater: ctx.inputs.keys():", ctx.inputs.keys())

    # # Add custom filters
    # for pattern, value in zip(ctx.inputs.keys(), custom_filter_values):
    #     if value is not None:  # Only process non-None values
    #         try:
    #             # Extract the JSON part before `.value`
    #             json_part = pattern.split('.')[0]
    #             id_dict = json.loads(json_part)  # Parse JSON into a dictionary
                
    #             # Extract "filter" key
    #             filter_name = id_dict.get("filter")
    #             if filter_name:  # If "filter" exists, add to kwargs
    #                 plot_generator_kwargs[filter_name] = value
    #                 print(f"Added to kwargs: {filter_name}: {value}")
    #             else:
    #                 print(f"Skipping key without 'filter': {id_dict}")
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON from pattern: {pattern}, Error: {e}")

    # # Final debug
    # print("Final plot_generator_kwargs:", plot_generator_kwargs)
    # print("---")


    # Add any custom filter values if they exist
    if custom_filter_values:
        # Get all input IDs
        input_ids = [
            key for key in ctx.inputs.keys() 
            if isinstance(json.loads(key.split('.')[0]), dict) and 
            json.loads(key.split('.')[0]).get("type") == "custom-filter"
        ]
        
        # Parse the pattern IDs to get filter names
        custom_filter_patterns = [json.loads(input_id.split('.')[0]) for input_id in input_ids]
        
        # Add custom filter values to kwargs using the filter name from the pattern
        for pattern, value in zip(custom_filter_patterns, custom_filter_values):
            if value is not None:  # Only add non-None values
                filter_name = pattern["filter"]  # This gets "city" from the pattern
                plot_generator_kwargs[filter_name] = value

    print("Updater: kwargs:", plot_generator_kwargs)

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
                                    html.H1("Nobel Laureate Data Dashboard v1.4", className="text-left mt-5 mb-5"),
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

                        dmc.TabsPanel(
                            html.Div(id="tab-content-overview"),  # Unique ID for tab content
                            value="tab_overview"
                        ),

                        dmc.TabsPanel(
                            children=html.Div(id="tab-content-current"),
                            value="tab_current"
                        ),

                       dmc.TabsPanel(
                            html.Div(id="tab-content-geography"),
                            value="tab_geography"
                        ),

                        dmc.TabsPanel(
                            html.Div(id="tab-content-demography"),
                            value="tab_demography"
                        ),

                        dmc.TabsPanel(
                            html.Div(id="tab-content-time"),
                            value="tab_time"
                        ),

                        dmc.TabsPanel(
                            html.Div(id="tab-content-migration"),
                            value="tab_migration"
                        ),

                        dmc.TabsPanel(
                            html.Div(id="tab-content-misc"),
                            value="tab_misc"
                        ),

                        dmc.TabsPanel(
                            html.Div(id="tab-content-data"),
                            value="tab_data"
                        ),
                    ],
                    value="tab_overview",  # Default selected tab
                    id="tabs"
                )
            ],
            fluid=True,
            style={"margin": "0px", "backgroundColor": "#ffffff", "maxWidth": "1200px"}
        )
    ]
)



##################################################################################################
# TAB CONTENT
##################################################################################################


##################################################################################################
# Tab Overview
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

@app.callback(
    Output("tab-content-overview", "children"),
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

        return content  # Return the content and set the pre-loading-trigger-status to TRUE, i.e. preloading can now start.

    else:
        return standard_loader_message


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
        Input("overview-timerange-control", "value")],
        # prevent_initial_call=True
)
def update_overview_content(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace, timerange):

    selected_categories = pdg.define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace)
    number_of_laureates, number_of_prizes, laureate_oldest_name, laureate_oldest_age, laureate_youngest_name, laureate_youngest_age, df_filtered_laureates = pdg.generate_overview_stats(selected_categories)

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
# Tab Current
##################################################################################################


@app.callback(
    Output('tab-content-current', 'children'),
    Input('tabs', 'value')
)
def render_tab_current_content(active_tab):
    if active_tab == 'tab_current':

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

                        
                        generate_loader_spinner("fig_sunburst_last"),

                        generate_loader_spinner("fig_map_movement"),

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
        return standard_loader_message



##################################################################################################
# Tab Geography
##################################################################################################

@app.callback(
    Output("tab-content-geography", "children"),  # Output for the tab content
    Input("tabs", "value"),  # Active tab
)
def render_tab_geography(active_tab):
    if active_tab == 'tab_geography':

        # Return the content for tab Nationality
        return dmc.Paper(
            children=[
                dmc.Stack(
                    children=[
                        
                        generate_loader_spinner("fig_choroplethglobe_prizespercountry"),
       
                        generate_loader_spinner("fig_map_cities"),

                        generate_loader_spinner("fig_bubbles_population"),

                        generate_loader_spinner("fig_bar_prizespercountry"),

                        generate_loader_spinner("fig_bar_prizespercountry_rs"),

                    ],
                    gap="lg"
                )

            ],
            shadow="md",
            radius="md",
            p="lg", 
            className="mt-3",
        )

    else:
        return standard_loader_message




##################################################################################################
# Tab Demography
##################################################################################################

@app.callback(
    Output('tab-content-demography', 'children'),
    Input('tabs', 'value')
)
def render_tab_demography_content(active_tab):
    if active_tab == 'tab_demography':
        # Return the content for tab Demography
        return dmc.Paper(
            children=[
                dmc.Stack(
                    children=[

                        generate_loader_spinner("fig_surface_prizesforwomen"),

                        generate_loader_spinner("fig_surface_prizesformenwomen"),

                        generate_loader_spinner("fig_donut_gender"),

                        generate_loader_spinner("fig_donut_ethnicity"),

                        generate_loader_spinner("fig_donut_religion"),

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
        return standard_loader_message


##################################################################################################
# Tab Time
##################################################################################################
@app.callback(
    Output('tab-content-time', 'children'),
    Input('tabs', 'value')
)
def render_tab_time_content(active_tab):
    if active_tab == 'tab_time':

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                dmc.Stack(
                    children=[

                        generate_loader_spinner("fig_histogram_timegap"),

                        generate_loader_spinner("fig_scatter_timegap_trend"),

                        generate_loader_spinner("fig_scatterbox_age"),
                        
                        generate_loader_spinner("fig_heatmap_age"),

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
        return standard_loader_message


##################################################################################################
# Tab Misc
##################################################################################################
@app.callback(
    Output('tab-content-misc', 'children'),
    Input('tabs', 'value')
)
def render_tab_misc_content(active_tab):
    if active_tab == 'tab_misc':    

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                dmc.Stack(
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

                        generate_loader_spinner("fig_line_prizemoney")
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
        return standard_loader_message



##################################################################################################
# Tab Migration
##################################################################################################

@app.callback(
    Output('tab-content-migration', 'children'),
    Input('tabs', 'value')
)
def render_tab_migration_content(active_tab):
    if active_tab == 'tab_migration':

        # Return the content for tab Time
        return dmc.Paper(
            children=[
                dmc.Stack(
                    children=[

                        generate_loader_spinner("fig_parcat_migration_dwp"),

                        generate_loader_spinner("fig_parcat_migration_bpd"),

                        generate_loader_spinner('fig_globe_movement')
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
        return standard_loader_message

##################################################################################################
# Tab Data
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

@app.callback(
    Output('tab-content-data', 'children'),
    Input('tabs', 'value')
)
def render_tab_data_content(active_tab):
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
        return standard_loader_message





##################################################################################################
# Running the app
##################################################################################################

# Run the app (locally)
# if __name__ == "__main__":
#     app.run(debug=True, port=5085) 

# # # Run the app on the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Fallback to port 8050 if PORT isn't set
    app.run_server(host='0.0.0.0', port=port, debug=False)
