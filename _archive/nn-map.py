import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import config as cf
    return cf, go, np, pl


@app.cell
def _(pl):
    df_nominations = pl.read_csv('nominations_full.csv', separator=';', encoding='utf-8')
    df_edges = pl.read_csv('df_edges.csv', separator=';', encoding='utf-8')
    df_match_edges_country = pl.read_csv('edges_country_match.csv', separator=';', encoding='utf-8')
    df_coordinates = pl.read_csv('countries_with_coordinates.csv', separator=';', encoding='utf-8')
    return df_coordinates, df_edges, df_match_edges_country, df_nominations


@app.cell
def _(df_nominations):
    df_nominations.head()
    return


@app.cell
def _(df_match_edges_country):
    df_match_edges_country.head()
    return


@app.cell
def _(df_edges, df_match_edges_country, pl):
    country_mapping = dict(zip(
        df_match_edges_country["CountryEdges"],
        df_match_edges_country["CountryRegular"]
    ))

    df_edges_2 = df_edges.with_columns(
        pl.col("nominator_country").replace(country_mapping),
        pl.col("nominee_country").replace(country_mapping)
    )
    return (df_edges_2,)


@app.cell
def _(df_edges_2, pl):
    gender_mapping = {"F":"female", "M":"male"}

    df_edges_3 = df_edges_2.with_columns(
        pl.col("nominator_gender").replace(gender_mapping),
        pl.col("nominee_gender").replace(gender_mapping)
    )
    return (df_edges_3,)


@app.cell
def _(df_edges_3):
    df_edges_3.head()
    return


@app.cell
def _(df_coordinates):
    df_coordinates.head()
    return


@app.cell
def _(df_coordinates, df_edges_3, pl):
    df_edges_4 = df_edges_3.join(
        df_coordinates.select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominator_lat"),
            pl.col("Longitude").alias("nominator_lon")
        ]),
        left_on="nominator_country",
        right_on="Country",
        how="left"
    )

    df_edges_4 = df_edges_4.join(
        df_coordinates.select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominee_lat"),
            pl.col("Longitude").alias("nominee_lon")
        ]),
        left_on="nominee_country",
        right_on="Country",
        how="left"
    )

    df_edges_4.head()
    return (df_edges_4,)


@app.cell
def _(df_edges_4):
    df_edges_4.shape
    return


@app.cell
def _(pl):
    # unverändert

    def filter_edges(data, categories="all", timerange_nomination=[1901, 1902], timerange_field="nomination", nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, nominee_name="", nominee_search_mode="all", nominee_gender="all", nominee_country="all", nominee_islaureate=False):

        # Replace short for categories handles with lists
        if categories == "all":
            categories=["Medicine", "Physics", "Chemistry", "Economic Sciences", "Literature", "Peace"]
        elif categories == "sci":
            categories=["Medicine", "Physics", "Chemistry", "Economic Sciences"]
        elif categories == "natsci":
            categories=["Medicine", "Physics", "Chemistry"]
        else:
            pass


        # Rewrite gender to format in list
        if nominator_gender == "female":
            nominator_gender = "F"
        elif nominator_gender == "male":
            nominator_gender = "M"
        else:
            pass

        # Rewrite gender to format in list
        if nominee_gender == "female":
            nominee_gender = "F"
        elif nominee_gender == "male":
            nominee_gender = "M"
        else:
            pass

        def is_not_empty(column_name):
            """Check if column is not empty (String or INT)"""
            return (
                ~pl.col(column_name).is_null() &
                ~pl.col(column_name).cast(pl.Utf8).str.strip_chars().is_in(["", "None", "NaN", "null"])
            )

        ### FILTER: CATEGORIES ###
        df_filtered = data.filter(pl.col("category").is_in(categories))

        ### FILTER: NOMINATOR GENDER ###
        if nominator_gender.lower() != "all":
            df_filtered = df_filtered.filter(pl.col("nominator_gender") == nominator_gender)
        else:
            pass

        ### FILTER: NOMINEE GENDER ###
        if nominee_gender.lower() != "all":
            df_filtered = df_filtered.filter(pl.col("nominee_gender") == nominee_gender)
        else:
            pass

        ### FILTER: NOMINATOR COUNTRY ###
        if nominator_country and nominator_country != "all" and len(nominator_country) > 0:
            if isinstance(nominator_country, str):
                nominator_country = [nominator_country]
            df_filtered = df_filtered.filter(
                pl.col("nominator_country").is_in(nominator_country) | 
                pl.col("nominator_country").is_null() | 
                (pl.col("nominator_country") == "Unknown")
            )

        ### FILTER: NOMINEE COUNTRY ###
        if nominee_country and nominee_country != "all" and len(nominee_country) > 0:
            if isinstance(nominee_country, str):
                nominee_country = [nominee_country]
            df_filtered = df_filtered.filter(
                pl.col("nominee_country").is_in(nominee_country) | 
                pl.col("nominee_country").is_null() | 
                (pl.col("nominee_country") == "Unknown")
            )

        ### FILTER: NOMINATOR IS LAUREATE ###
        if nominator_islaureate:
            df_filtered = df_filtered.filter(is_not_empty("nominator_prizes"))

        ### FILTER: NOMINEE IS LAUREATE ###
        if nominee_islaureate:
            df_filtered = df_filtered.filter(is_not_empty("nominee_prizes"))

        ### FILTER: TIMERANGE ###
        if timerange_nomination is not None:
            df_filtered = df_filtered.filter(
                pl.col("year").is_between(timerange_nomination[0], timerange_nomination[1]) | (pl.col("year")==0)
            )


        ### FILTER: NOMINATOR NAME ###
        def search_in_columns_simple(search_terms, columns, mode="any"):

            # Concatenate all columns into a single string column
            combined_text = pl.concat_str([
                pl.col(col).cast(pl.Utf8).fill_null("") for col in columns
            ], separator=" ")

            if mode.lower() == "any":
                # At least one term must be present
                conditions = [
                    combined_text.str.contains(f"(?i){term}") 
                    for term in search_terms
                ]
                return pl.any_horizontal(conditions)

            elif mode.lower() == "all":
                # All terms must be present
                conditions = [
                    combined_text.str.contains(f"(?i){term}") 
                    for term in search_terms
                ]
                return pl.all_horizontal(conditions)

        # Only filter if there is a nominator name input
        if nominator_name and len(nominator_name) > 0:
            columns = ['nominator_name']
            df_filtered = df_filtered.filter(search_in_columns_simple(nominator_name, columns, mode=nominator_search_mode))
        else:
            pass 

        # Only filter if there is a nominee name input
        if nominee_name and len(nominee_name) > 0:
            columns = ['nominee_name']
            df_filtered = df_filtered.filter(search_in_columns_simple(nominee_name, columns, mode=nominee_search_mode))
        else:
            pass 

        return df_filtered
    return (filter_edges,)


@app.cell
def _(df_edges_4, filter_edges):
    df_edges_filtered = filter_edges(df_edges_4)
    df_edges_filtered.head()
    return (df_edges_filtered,)


@app.cell
def _(df_edges_filtered):
    df_edges_filtered.shape
    return


@app.cell
def _(cf, df_edges, filter_edges, go, np, pl):
    def generate_map_movement(data=df_edges, categories="all", timerange_nomination=[1901, 1902], timerange_field="nomination", nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, nominee_name="", nominee_search_mode="all", nominee_gender="all", nominee_country="all", nominee_islaureate=False):

        data = filter_edges(data, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    

        data = data.filter(
            pl.col("nominator_lat").is_not_null() &
            pl.col("nominator_lon").is_not_null() &
            pl.col("nominee_lat").is_not_null() &
            pl.col("nominee_lon").is_not_null()
        ).with_columns([
            pl.col("nominator_lat").cast(pl.Float64),
            pl.col("nominator_lon").cast(pl.Float64),
            pl.col("nominee_lat").cast(pl.Float64),
            pl.col("nominee_lon").cast(pl.Float64)
        ])

        # Function to interpolate points for curved paths
        def interpolate_points(lat1, lon1, lat2, lon2, num_points=50):
            """
            Generates intermediate lat/lon points for a geodesic (great circle) path.
            """
            lats = []
            lons = []

            # Convert lat/lon to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

            # Calculate differences
            d = 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
            f = np.linspace(0, 1, num_points)

            # Interpolate points
            for t in f:
                if np.isclose(d, 0):
                    # For very small or zero distances, use a straight path or set the curve to a straight line
                    A = 1 - t
                    B = t
                else:
                    # Otherwise, use the original formula
                    A = np.sin((1 - t) * d) / np.sin(d)
                    B = np.sin(t * d) / np.sin(d)
                x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
                y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
                z = A * np.sin(lat1) + B * np.sin(lat2)
                lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
                lon = np.arctan2(y, x)
                lats.append(np.degrees(lat))
                lons.append(np.degrees(lon))

            return lats, lons

        # Initialize the figure
        fig = go.Figure()

        # Add markers for nominator cities
        fig.add_trace(go.Scattermap(
            lon=data["nominator_lon"],
            lat=data["nominator_lat"],
            text=data["nominator_name"],
            hoverinfo="text",
            mode="markers",
            marker=go.scattermap.Marker(
                size=10,
                color=cf.c_teal,
                opacity=0.7
            ),
            name="Nominator"
        ))

        # Add markers for nominee cities
        fig.add_trace(go.Scattermap(
            lon=data["nominee_lon"],
            lat=data["nominee_lat"],
            text=data["nominee_name"],
            hoverinfo="text",
            mode="markers",
            marker=go.scattermap.Marker(
                size=10,
                color=cf.c_teal,
                opacity=0.9
            ),
            name="Nominee"
        ))

        # Add migration paths for each laureate with curvature
        # colors = [cf.c_brown, cf.c_darkmagenta, cf.c_lightblue, cf.c_orange, cf.c_pink, cf.c_red, cf.c_teal]
        colors = cf.c_colorscale_palette
        for i, row in enumerate(data.iter_rows(named=True)):
            color = colors[i % len(colors)]  # Cycle through the colors list

            # Calculate interpolated points for curvature
            lats, lons = interpolate_points(
                row["nominator_lat"], row["nominator_lon"],
                row["nominee_lat"], row["nominee_lon"],
                num_points=100
            )

            # Add a trace for each migration path
            fig.add_trace(go.Scattermap(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=3, color=color),
                opacity=0.7,
                hoverinfo="text",
                text=f"Nominator: {row['nominator_name']}<br>Country: {row['nominator_country']}<br>Nominee: {row['nominee_name']}<br>Country:{row['nominee_country']}",
                name=f"{row['nominator_name']}"
            ))

        # Update layout for the mapbox visualization
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor=cf.c_plot_background,
            title_text="Countries of Nominator & Nominee",
            showlegend=False,
            mapbox=dict(
                style="carto-positron",  # Other styles: "streets", "dark", "light", "satellite", etc.
                center=dict(lat=30, lon=0),  # Center the map globally
                zoom=0.8,

            ),

            hoverlabel=dict(
                    bgcolor=cf.c_hoverlabel_bg,
                    font_size=12,
                    font_family="Rubik"
            ),
            # height=800,
            margin=dict(l=0, r=0, t=0, b=0),  # Reduce the margins
        )

        return fig
    return (generate_map_movement,)


@app.cell(disabled=True)
def _(generate_map_movement):
    fig = generate_map_movement()
    fig.show()
    return


@app.cell
def _(cf, df_edges_filtered, go, np, pl):
    def generate_map_movement_animate1(data=df_edges_filtered):

        data = data.filter(
            pl.col("nominator_lat").is_not_null() &
            pl.col("nominator_lon").is_not_null() &
            pl.col("nominee_lat").is_not_null() &
            pl.col("nominee_lon").is_not_null()
        ).with_columns([
            pl.col("nominator_lat").cast(pl.Float64),
            pl.col("nominator_lon").cast(pl.Float64),
            pl.col("nominee_lat").cast(pl.Float64),
            pl.col("nominee_lon").cast(pl.Float64)
        ]).sort("year")

        # Function to interpolate points for curved paths
        def interpolate_points(lat1, lon1, lat2, lon2, num_points=50):
            lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
            lats = []
            lons = []
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            d = 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
            f = np.linspace(0, 1, num_points)

            for t in f:
                if np.isclose(d, 0):
                    A = 1 - t
                    B = t
                else:
                    A = np.sin((1 - t) * d) / np.sin(d)
                    B = np.sin(t * d) / np.sin(d)
                x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
                y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
                z = A * np.sin(lat1) + B * np.sin(lat2)
                lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
                lon = np.arctan2(y, x)
                lats.append(np.degrees(lat))
                lons.append(np.degrees(lon))

            return lats, lons

        fig = go.Figure()

        # Add markers for nominator cities
        fig.add_trace(go.Scattermap(  # Geändert von Scattermapbox
            lon=data["nominator_lon"],
            lat=data["nominator_lat"],
            text=data["nominator_name"],
            hoverinfo="text",
            mode="markers",
            marker=dict(size=10, color=cf.c_teal, opacity=0.7),  # Als dict!
            name="Nominator"
        ))

        # Add markers for nominee cities
        fig.add_trace(go.Scattermap(  # Geändert von Scattermapbox
            lon=data["nominee_lon"],
            lat=data["nominee_lat"],
            text=data["nominee_name"],
            hoverinfo="text",
            mode="markers",
            marker=dict(size=10, color=cf.c_teal, opacity=0.9),  # Als dict!
            name="Nominee"
        ))

        # Add migration paths
        colors = cf.c_colorscale_palette

        for i, row in enumerate(data.iter_rows(named=True)):
            color = colors[i % len(colors)]

            lats, lons = interpolate_points(
                row["nominator_lat"], row["nominator_lon"],
                row["nominee_lat"], row["nominee_lon"],
                num_points=100
            )

            fig.add_trace(go.Scattermap(  # Geändert von Scattermapbox
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=5, color=color),
                opacity=0.7,
                hoverinfo="text",
                text=f"{row['nominator_name']}<br>Country: {row['nominator_country']}<br>Nominee: {row['nominee_country']}",
                name=f"{row['nominator_name']}"
            ))

        fig.update_layout(
            template='plotly_white',
            plot_bgcolor=cf.c_plot_background,
            title_text="Countries of Nominator & Nominee",
            showlegend=False,
            map=dict(  # Geändert von mapbox zu map
                style="open-street-map",  # Oder "carto-positron", "white-bg"
                center=dict(lat=30, lon=0),
                zoom=0.8,
            ),
            hoverlabel=dict(
                bgcolor=cf.c_hoverlabel_bg,
                font_size=12,
                font_family="Rubik"
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig
    return (generate_map_movement_animate1,)


@app.cell
def _(generate_map_movement_animate1):
    fig2 = generate_map_movement_animate1()
    fig2.show()
    return


if __name__ == "__main__":
    app.run()
