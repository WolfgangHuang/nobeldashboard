"""
Migration figures: parallel categories, movement globe/map.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from data import df_laureates, lastyearincluded
from filters import standard_filter


def generate_parcat_migration(data=df_laureates, loc1="ParCatDegreeCountry", loc2="ParCatWorkCountry", loc3="ParCatPrizeCountry", categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", width=1000, height=1400):
    """
    Generates a parallel categories diagram visualizing Nobel laureates' movement 
    via three locations (Degree/WorkPrize or Birth/Prize/Death)

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        
        For Degree/WorkPrize:
        loc1 (str): Column representing the first location (default: "ParCatDegreeCountry").
        loc2 (str): Column representing the second location (default: "ParCatWorkCountry").
        loc3 (str): Column representing the third location (default: "ParCatPrizeCountry").
        width (int): plot width (default: 1600)
        height (int): plot height (default: 1400)
        
        For Birth/Prize/Death:
        pass
        loc1="BirthCountryNow", loc2="Prize0_Affiliation0_Country", loc3="DeathCountryNow"
        width=1400, height=1800
        
        categories (str): categories filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').

    Returns:
        plotly.graph_objects.Figure: A parallel categories diagram showing 
        movement paths, with customizable dimensions and color-coded categories.
    """

    data = standard_filter(data, categories, gender, timerange, timerange_field)


    # .copy() to ensure we don't modify the original dataframe
    data = data[data['ParCatDegreeCountry'].notna() & (data['ParCatDegreeCountry'].str.strip() != "")].copy()

    data.loc[:,'color_value'] = data['ParCatPrizeCountry'].factorize()[0]  # Factorize converts categories to unique integers

    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            {'label': 'Degree', 'values': data[loc1]},
            {'label': 'Work', 'values': data[loc2]},
            {'label': 'Prize', 'values': data[loc3]}
        ],
        line={
            'color': data['color_value'],  # Use the mapped numerical values for coloring
            'colorscale': cf.c_colorscale_palette_light,
            'shape': 'hspline'  # hspline is the attribute for curved lines
        },
        hoveron='category', # Hover on color
        hoverinfo='all', # Display all available information on hover
        arrangement='freeform', # Allows for dragging categories without snapping to a grid
    )])

    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
        ),
        # title=dict(
        #     text = "4.a: Movement from Locations of Degree / Achievement / Prize",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        width = width,
        height = height,
        margin=dict(l=60, r=60, t=30, b=10)
        ),


    # Hover label styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )

    return fig


def prepare_data_splines(data, year, categories, gender, timerange, timerange_field):
   """
   Filters and formats Nobel laureates' data for visualizing movement splines.

   Parameters:
   ----------
   data : pd.DataFrame
      Input DataFrame with laureates' details.
   year : str or int
      Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work.
   categories : str
      Nobel Prize categories (e.g., 'physics', 'all').
   gender : str
      Gender filter ('male', 'female', 'all').

   Returns:
   -------
   pd.DataFrame
      Processed DataFrame with key columns and valid location data.
   """

   # standard filter
   data = standard_filter(data, categories, gender, timerange, timerange_field)
   # get required columns
   df_movement_splines = data[["AwardeeDisplayName", "Prize0_AwardYear", "BirthCityNow", "BirthCountryNow","BirthContinent", "BirthCityNowLat", "BirthCityNowLon",
      "Prize0_Affiliation0_CityNow", "Prize0_Affiliation0_Country","Prize0_Affiliation0_Continent", "Prize0_Affiliation0_CityLatitude", "Prize0_Affiliation0_CityLongitude", "DeathCityNow", "DeathCountryNow", "DeathContinent", "DeathCityLat", "DeathCityLon"]]
   
   # fill na with none
   df_movement_splines = df_movement_splines.fillna('None')

   # filter out none
   df_movement_splines = df_movement_splines[
    (df_movement_splines["BirthCityNowLat"] != 'None') & 
    (df_movement_splines["BirthCityNowLon"] != 'None') &
    (df_movement_splines["Prize0_Affiliation0_CityLatitude"] != 'None') &
    (df_movement_splines["Prize0_Affiliation0_CityLongitude"] != 'None')
   ]

   # filter for year parameter
   if year=="last":
      df_filtered = df_movement_splines[df_movement_splines["Prize0_AwardYear"] == lastyearincluded]
   else:
      df_filtered = df_movement_splines

   return df_filtered


def generate_globe_movement(data=df_laureates, year="all", categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Creates a globe visualization showing Nobel laureates' migration paths.

    Parameters:
    ----------
    data : pd.DataFrame, optional
        Input DataFrame containing laureates' data (default is `df_laureates`).
    year : str or int, optional
        Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work. (default is "last").
    categories : str, optional
        Nobel Prize categories to filter (e.g., 'physics', default is "all").
    gender : str, optional
        Gender filter ('male', 'female', default is "all").

    Returns:
    -------
    plotly.graph_objs._figure.Figure
        A Plotly globe visualization with migration paths and birth city markers.
    """

    # Sample colors (you can customize this list as needed or use a color scale)
    # colors = [
    # "red", "blue", "green", "orange", "purple", "brown", "pink", "cyan", "magenta", "yellow"
    # ]

    colors = cf.c_colorscale_palette

    data = prepare_data_splines(data, year, categories, gender, timerange, timerange_field)

    # Initialize the figure
    fig = go.Figure()

    # Add markers for birth cities
    fig.add_trace(go.Scattergeo(
        lon=data["BirthCityNowLon"],
        lat=data["BirthCityNowLat"],
        hoverinfo="text",
        text=data["AwardeeDisplayName"],
        mode="markers",
        marker=dict(
            size=4,
            color=th.FIG_INK_LIGHT,  # ink markers; swap to light in dark mode
            line=dict(width=0.5, color="rgba(68, 68, 68, 0)")
        ),
        name="Birth Cities"
    ))

    # Add migration paths for each laureate
    for i, row in data.iterrows():
        color = colors[int(i) % len(colors)]  # Cycle through the colors list

        # Add a trace for each migration path
        fig.add_trace(go.Scattergeo(
            lon=[row["BirthCityNowLon"], row["Prize0_Affiliation0_CityLongitude"], None],
            lat=[row["BirthCityNowLat"], row["Prize0_Affiliation0_CityLatitude"], None],
            mode="lines",
            line=dict(width=1, color=color),
            opacity=0.6,
            hoverinfo="text",
            text=f"<b>{row['AwardeeDisplayName']}</b><br>Birth: {row['BirthCityNow']}<br>Affiliation: {row['Prize0_Affiliation0_CityNow']}",
            name=f"{row['AwardeeDisplayName']}"
        ))

    # Update layout for global view
    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        # title_text="Migration Paths of 2023 Nobel Prize Laureates",
        showlegend=False,
        geo=go.layout.Geo(
            projection_type="orthographic",
            showland=True,
            countrycolor=th.GEO_LIGHT["stroke"],
            countrywidth=0.8,  # Border width
            coastlinecolor=th.GEO_LIGHT["stroke"],
            coastlinewidth=0.5,  # Coastline width
            showlakes=True,
            showcountries=True,
            showocean=True,
            showframe=False,  # Removes the box frame
            bgcolor=th.GEO_LIGHT["frame"],
            landcolor=th.GEO_LIGHT["land"],
            oceancolor=th.GEO_LIGHT["water"],
            rivercolor=th.GEO_LIGHT["water"],
            lakecolor=th.GEO_LIGHT["water"],
        ),
        height=900,
        #autosize=True
    )

    # Hover label styling comes from the theme template (theme-aware surface color).

    return fig


def generate_map_movement(data=df_laureates, year="last", categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Creates an interactive map visualizing Nobel laureates' migration paths with curved lines.

    Parameters:
    ----------
    data : pd.DataFrame, optional
        Input DataFrame containing laureates' data (default is `df_laureates`).
    year : str or int, optional
        Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work. (default is "last").
    categories : str, optional
        Nobel Prize categories to filter (e.g., 'physics', default is "all").
    gender : str, optional
        Gender filter ('male', 'female', default is "all").

    Returns:
    -------
    plotly.graph_objs._figure.Figure
        A Plotly mapbox visualization with migration paths, birth, and affiliation markers.
    """
    
    # colors = colorscale_palette

    data = prepare_data_splines(data, year, categories, gender, timerange, timerange_field)


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

    # Add markers for birth cities
    fig.add_trace(go.Scattermapbox(
        lon=data["BirthCityNowLon"],
        lat=data["BirthCityNowLat"],
        text=data["AwardeeDisplayName"],
        hoverinfo="text",
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=10,
            color=th.SPECTRUM_LIGHT["Peace"],  # spectrum green; swaps to Neon in dark mode
            opacity=0.7
        ),
        name="Birth Cities"
    ))

    # Add markers for affiliation cities
    fig.add_trace(go.Scattermapbox(
        lon=data["Prize0_Affiliation0_CityLongitude"],
        lat=data["Prize0_Affiliation0_CityLatitude"],
        text=data["AwardeeDisplayName"],
        hoverinfo="text",
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=10,
            color=th.SPECTRUM_LIGHT["Peace"],  # spectrum green; swaps to Neon in dark mode
            opacity=0.9
        ),
        name="Affiliation Cities"
    ))

    # Add migration paths for each laureate with curvature
    colors = th.colorway(False)  # spectrum colors; swap to Neon in dark mode
    for i, row in data.iterrows():
        color = colors[int(i) % len(colors)]  # Cycle through the colors list

        # Calculate interpolated points for curvature
        lats, lons = interpolate_points(
            row["BirthCityNowLat"], row["BirthCityNowLon"],
            row["Prize0_Affiliation0_CityLatitude"], row["Prize0_Affiliation0_CityLongitude"],
            num_points=100
        )

        # Add a trace for each migration path
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=5, color=color),
            opacity=0.7,
            hoverinfo="text",
            text=f"{row['AwardeeDisplayName']}<br>Birth: {row['BirthCityNow']}<br>Affiliation: {row['Prize0_Affiliation0_CityNow']}",
            name=f"{row['AwardeeDisplayName']}"
        ))

    # Update layout for the mapbox visualization
    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        title_text="Places of Birth & Affiliation",
        showlegend=False,
        mapbox=dict(
            style=th.MAPBOX_STYLE_LIGHT,  # swaps to carto-darkmatter in dark mode
            center=dict(lat=30, lon=0),  # Center the map globally
            zoom=0.8,
        ),
        # hoverlabel styling comes from the theme template (theme-aware)
        # height=800,
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce the margins
    )

    return fig
