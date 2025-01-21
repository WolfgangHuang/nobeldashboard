



# plot layout geenrator without classes
def generate_plot_in_layout(
    cols={"base": 1, "sm": 1},
    header="Generic Plot Title",
    subheader="",
    datafrom="1901",
    datato=lastyearincluded,
    badges=[dmc.Badge("All Categories", variant="outline", color=brand_color_alt)],
    plot_generator=None,
    plot_generator_kwargs={},
    plot_id="fig_type_content",
    figure=None,
    style={'width': '100%', 'height': '100%'},
    footer="",
    content_classname="widget-content",
    show_filters={"categories": True, "gender": True, "custom-filter": None},
):
    # Debug print
    #print(f"Generator: Generating plot with ID: {plot_id} and plot_generator: {plot_generator} and kwargs: {plot_generator_kwargs}")

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

                    # Badges
                    dmc.Stack(
                        children=[
                            html.Div(badges)
                        ],
                        style={
                            "marginBottom": "10px",
                        }
                    ),

                    # Filter Modal
                    html.Div(
                        [
                            dmc.Button("Filter", variant="gradient", gradient={"from": c_lightblue, "to": c_teal}, size="xs", id={"type": "filter-button", "index": plot_id}),
                            dmc.Modal(
                                title="Filter",
                                centered=True,
                                id={"type": "filter-modal", "index": plot_id},
                                size="80%",
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
                                                style={
                                                    "marginRight": "30px",
                                                    "display": "block" if show_filters["categories"] else "none"
                                                }
                                            ),
                                            # Gender Filter
                                            dmc.Stack(
                                                children=[
                                                    html.Div("Gender"),
                                                    html.Div(
                                                        dmc.Group(
                                                            [
                                                                dmc.Chip("female", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-female", "index": plot_id}),
                                                                dmc.Chip("male", size="xs", variant="outline", checked=True, color=c_brown, id={"type": "chip-male", "index": plot_id}),
                                                            ]
                                                        )
                                                    ),
                                                ],
                                                style={
                                                    "display": "block" if show_filters["gender"] else "none"
                                                }
                                            ),

                                            # Custom Filter (if provided)
                                            html.Div(
                                                show_filters["custom-filter"],
                                                style={
                                                    "display": "block" if show_filters["custom-filter"] else "none"
                                                }
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
                    # dcc.Store(id={"type": "plot-generator", "index": plot_id}, data=plot_generator),
                    # dcc.Store(id={"type": "plot-generator-kwargs", "index": plot_id}, data=plot_generator_kwargs)
                ],
                className="widget-container",
            ),
        ],
    )




# Generate plots and save to pickle file, from precompute_plots.py

# Generate plots
    def generate_plots():

        # Tab Overview
        fig_donut_gender = generate_donut(df_laureates, characteristic="gender")
        fig_donut_ethnicity = generate_donut(df_laureates, characteristic="ethnicity")
        fig_donut_religion = generate_donut(df_laureates, characteristic="religion")
        fig_sunburst = generate_sunburst(df_laureates)
        
        # Tab Current
        fig_sunburst_last = generate_sunburst(df_laureates, year="last")
        fig_map_movement = generate_map_movement(df_laureates, year="last")

        # Tab Geography
        fig_choroplethglobe_prizespercountry = generate_choroplethglobe(df_laureates)
        fig_map_cities = generate_scattermapbox_cities(df_laureates)
        fig_bubbles_population = generate_bubbles_perpopulation(df_prizes)
        fig_bar_prizespercountry = generate_bar_percountry(df_prizes)
        fig_bar_prizespercountry_rs = generate_bar_percountry(df_prizes, runningsum=True)
        
        # Tab Demography
        fig_surface_prizesforwomen = generate_3dsurface_pergender(df_prizes, gender="female")
        fig_surface_prizesformenwomen = generate_3dsurface_pergender(df_prizes, gender="all")

        # Tab Time
        fig_histogram_timegap = generate_histogram_timegap(df_prizes, categories="natsci")
        fig_scatter_timegaptrend = generate_scatterbox_timegaptrend(df_prizes, categories="natsci")
        fig_scatterbox_age = generate_scatterbox_age(df_laureates)
        fig_heatmap_age = generate_heatmap_age(df_laureates)

        # Tab Migration
        fig_parcat_migration_dwp = generate_parcat_migration(df_laureates)
        fig_parcat_migration_bpd = generate_parcat_migration(df_laureates, loc1="BirthCountryNow", loc2="Prize0_Affiliation0_Country", loc3="DeathCountryNow", width=1400, height=1800)
        fig_globe_movement = generate_globe_movement(df_laureates)

        # Tab Misc
        fig_line_prizemoney, totalprizeamount = generate_line_prizemoney(df_prizes)

        return {

        # Tab Overview
        'fig_donut_gender': fig_donut_gender,
        'fig_donut_ethnicity': fig_donut_ethnicity,
        'fig_donut_religion': fig_donut_religion,
        'fig_sunburst': fig_sunburst,

        # Tab Current
        'fig_sunburst_last': fig_sunburst_last,
        'fig_map_movement': fig_map_movement,

        # Tab Geography
        'fig_choroplethglobe_prizespercountry' : fig_choroplethglobe_prizespercountry,
        'fig_map_cities' : fig_map_cities,
        'fig_bubbles_population' : fig_bubbles_population,
        'fig_bar_prizespercountry' : fig_bar_prizespercountry,
        'fig_bar_prizespercountry_rs' : fig_bar_prizespercountry_rs,

        # Tab Demography
        'fig_surface_prizesforwomen' : fig_surface_prizesforwomen,
        'fig_surface_prizesformenwomen' : fig_surface_prizesformenwomen,

        # Tab Time
        'fig_histogram_timegap' : fig_histogram_timegap,
        'fig_scatter_timegaptrend' : fig_scatter_timegaptrend,
        'fig_scatterbox_age' : fig_scatterbox_age,
        'fig_heatmap_age' : fig_heatmap_age,

        # Tab Migration
        'fig_parcat_migration_dwp': fig_parcat_migration_dwp,
        'fig_parcat_migration_bpd': fig_parcat_migration_bpd,
        'fig_globe_movement' : fig_globe_movement,

        # Tab Misc
        'fig_line_prizemoney' : fig_line_prizemoney,
        'totalprizeamount' : totalprizeamount
        }
 

    # Save plots to a pickle file

    # Start timing
    start_time = time.time()

    # Generate the plots
    pcp_plots = generate_plots()

    # Save to a pickle file
    pickle_file = 'pcp_plots.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(pcp_plots, f)

    # Measure elapsed time
    elapsed_time = time.time() - start_time

    # Get the file size
    file_size = os.path.getsize(pickle_file) / (1024 * 1024)  # Convert bytes to MB

    print(f"All plots precomputed in {elapsed_time:.2f} seconds.")
    print(f"Pickle file size: {file_size:.2f} MB.")


