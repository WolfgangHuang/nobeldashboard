"""
PlotConfig class + the per-plot configuration data for every dashboard widget.
Extracted from config.py (which keeps app settings and color constants).
"""

from dash import dcc, html
import dash_mantine_components as dmc


class PlotConfig:
    """
    Configuration container for dashboard plots.
    
    Centralizes all plot settings including display options, filter configurations,
    and generator function references. Each plot in the dashboard has a corresponding
    PlotConfig instance in the plot_configs dictionary.
    
    Attributes:
        plot_id (str): Unique identifier for the plot (e.g., 'fig_choroplethglobe').
        lastyearincluded (int): Most recent year in the dataset.
        plot_category (str): 'standard' or 'nominations' - determines layout generator.
        header (str): Plot title displayed in the widget header.
        subheader (str): Descriptive text below the title.
        data_range (tuple): (start_year, end_year) for display in badges.
        show_filters (dict): Which filter controls to show:
            - categories (bool): Category chips (Medicine, Physics, etc.)
            - gender (bool): Gender chips (Male, Female)
            - timerange (bool): Year range slider
            - custom-filter (str|None): Custom filter component name
            - algorithm (bool): Network layout algorithm dropdown
        badges (list[str]): Text for filter state badge indicators.
        filter_type (str): 'standard' or 'nominations'.
        chips_notchecked (list[str]): Category chips unchecked by default.
        chips_disabled (list[str]): Category chips that cannot be toggled.
        timerange (list): [start_year, end_year] default slider values.
        timerange_field (str): Date field for filtering - 'award', 'birth', 'death'.
        plot_generator (str): Function name in plotdatagenerator module.
        plot_generator_kwargs (dict): Default arguments for the generator function.
        footer (list): Optional footer content components.
        style (dict): CSS styles for the plot container.
        
    Example:
        >>> config = PlotConfig(
        ...     plot_id='fig_bar_example',
        ...     lastyearincluded=2024,
        ...     header='Prizes by Country',
        ...     plot_generator='generate_bar_percountry',
        ...     plot_generator_kwargs={'country': 'birth'}
        ... )
    """
    
    def __init__(
        self,
        plot_id,
        lastyearincluded,
        plot_category=None,
        header=None,
        subheader=None,
        data_range=None,
        show_filters=None,
        badges=None,
        filter_type=None,
        chips_notchecked=None,
        chips_disabled=None,
        timerange=None,
        timerange_field=None,
        plot_generator=None,
        plot_generator_kwargs=None,
        footer=None,
        style=None
    ):
        self.plot_id = plot_id
        self.lastyearincluded = lastyearincluded
        self.plot_category = plot_category or "standard"
        self.header = header or "Generic Plot Title"
        self.subheader = subheader
        self.data_range = data_range or ("1901", lastyearincluded)
        self.show_filters = show_filters or {"categories": True, "gender": True, "timerange": True, "custom-filter": None, "algorithm":True}
        self.badges = badges or ["All Categories", f"1901-{lastyearincluded}"]
        self.filter_type = filter_type or "standard"
        self.chips_notchecked = chips_notchecked or []
        self.chips_disabled = chips_disabled or []
        self.timerange = timerange or ["1901", lastyearincluded]
        self.timerange_field = timerange_field or "award"
        self.plot_generator = plot_generator
        self.plot_generator_kwargs = plot_generator_kwargs or {}
        self.footer = footer or []
        self.style = style or {'width': '100%', 'height': '100%'}

    def get_plot_generator(self, pdg_module):
        """
        Retrieve the plot generator function from the plotdatagenerator module.
        
        Args:
            pdg_module: The plotdatagenerator module instance.
            
        Returns:
            callable: The generator function for this plot.
            
        Raises:
            ValueError: If plot_generator attribute is not set.
        """
        if not self.plot_generator:
            raise ValueError("Plot generator function not defined.")
        return getattr(pdg_module, self.plot_generator, None)

    def generate_layout(self):
        """
        Generate the complete widget layout for this plot.
        
        Delegates to the appropriate layout generator based on plot_category:
        - 'nominations': Uses generate_plot_in_layout_nominations()
        - 'standard' (default): Uses generate_plot_in_layout_standard()
        
        Returns:
            dash component: Complete plot widget with header, controls, and figure.
        """
        if self.plot_category == "nominations":
            from layouts import generate_plot_in_layout_nominations
            return generate_plot_in_layout_nominations(self)
        else:
            from layouts import generate_plot_in_layout_standard
            return generate_plot_in_layout_standard(self)

    def generate_badges(self):
        """
        Create badge components showing current filter state.
        
        Generates dmc.Badge components for each item in self.badges,
        typically showing category and time range information.
        
        Returns:
            list[dmc.Badge]: List of badge components for the header.
        """
        badges_code = []
        for badge in self.badges:
            badges_code.append(dmc.Badge(badge, variant="outline", color="economics", mr="xs"))
        return badges_code

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
# network
#   - fig_network_nominations
# # networkmap


def get_plot_configs(df_laureates, df_prizes, df_nominations, lastyearincluded):
    """
    Generate plot configurations dictionary.
    
    Args:
        df_laureates: DataFrame with laureates data
        df_prizes: DataFrame with prizes data
    
    Returns:
        dict: Plot configurations
    """
    from plotdatagenerator import lastyearincluded, totalprizeamount
    from dash import dcc
    
    # lastyearincluded = pdg.get_lastyearincluded(df_prizes)
    # totalprizeamount = pdg.generate_var_prizeamount(df_prizes)

   
    return {

    # Choropleth Globe
    "fig_choroplethglobe_prizespercountry": PlotConfig(
        plot_id="fig_choroplethglobe_prizespercountry",
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
        header = "Laureate Age at Time of Award",
        subheader = "This is basically the same data, but presented differently. Here, you see the time gap for all prizes (averaged in case of multiple winners) in all years. The plot also shows the trendlines (going up), as well as the average life expectancy (also going up).",
        plot_generator="generate_scatterbox_age",
        plot_generator_kwargs={"data": df_laureates},
        footer = [dcc.Markdown("**Interesting Findings**: In the early years, the average age in the natural sciences was around 45, whereas nowadays it is close to 65. This fits well to the earlier finding that the timegap has increased by - on average - 25 years. Interestingly enough, peace prize awardees get younger.")],
    ),

    # Heatmap Age
    "fig_heatmap_age": PlotConfig(
        plot_id="fig_heatmap_age",
        lastyearincluded=lastyearincluded,
        header = "Laureate Age at Time of Award (Heatmap)",
        subheader = "Same data as above, but displayed as heatmap.",
        plot_generator="generate_heatmap_age",
        plot_generator_kwargs={"data": df_laureates},
        footer=[],
    ),

    # Parcat Migration
    "fig_parcat_migration_dwp": PlotConfig(
        plot_id="fig_parcat_migration_dwp",
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
        header="Discipline - Gender - Country",
        subheader="Click on the segments to filter the data.",
        badges=["All Categories", f"{lastyearincluded}"],
        plot_generator="generate_sunburst",
        plot_generator_kwargs={"data": df_laureates, "year": "last"},
        footer=[
            dcc.Markdown("**Interesting Findings**: Let's see who will receive the remaining prizes.")
        ],
    ),

    # Globe Movement
    "fig_globe_movement": PlotConfig(
        plot_id="fig_globe_movement",
        lastyearincluded=lastyearincluded,
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
        lastyearincluded=lastyearincluded,
        header="Life Paths (Birth - Work)",
        subheader="Some laureates haven't moved and are represented as dots.",
        badges=["All Categories", f"{lastyearincluded}"],
        plot_generator="generate_map_movement",
        plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: Let's see who will receive the remaining prizes.")
        ],
    ),

    # Most Common Firstnames
    "fig_mostcommon_firstnames": PlotConfig(
        plot_id="fig_mostcommon_firstnames",
        lastyearincluded=lastyearincluded,
        header="Most Common Firstnames",
        subheader="What to name your kid if you want it to become a Nobel laureate.",
        plot_generator="generate_mostcommon_firstnames",
        #plot_generator_kwargs={"data": df_laureates},
        footer=[
            dcc.Markdown("**Interesting Findings**: We may assume that the name itself will not have much of an influence on the laureates' success. However, the names are a good indicator of the social and economic status, as well as nationality and of course gender.")
        ],
    ),

    # Nominations Network
    "fig_network_nominations": PlotConfig(
        plot_id="fig_network_nominations",
        lastyearincluded=lastyearincluded,
        plot_category="nominations",
        header = "Graph/Network of Nominations",
        subheader = f"Shows a graph visualization of nominators, nominations and nominees for the Nobel prizes. The initial graph shows all female nominees until {lastyearincluded - 49}. You may zoom and pan the graph. Hovering over a node shows additional information. Clicking on a node highlights edges and shows detail information below the graph. Use the filters to customize the view.",
        badges=["All Categories", f"1901-{lastyearincluded -50}"],
        filter_type="nominations",
        plot_generator_kwargs={"data":df_nominations},
        plot_generator = "generate_network",
        footer = [
            dcc.Markdown("")
        ]
    ),

    # Nominations Network Map
    "fig_map_nominations": PlotConfig(
        plot_id="fig_map_nominations",
        lastyearincluded=lastyearincluded,
        plot_category="nominations",
        header = "World Map of Nominations",
        subheader = f"Nominations visualized as splines of world map, showing the flow from nominator to nominee. The initial graph shows all female nominees until 1976. You may zoom and pan the graph.",
        badges=["All Categories", f"1901-{lastyearincluded -50}"],
        filter_type="nominations",
        show_filters={"categories": True, "gender": True, "custom-filter": False, "algorithm": False},
        plot_generator_kwargs={"data":df_nominations},
        plot_generator = "generate_map_nominations",
        footer = [
            dcc.Markdown("")
        ]
    ),

}
