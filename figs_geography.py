"""
Geography figures: choropleth globe, city map, population bubbles, country bars.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from data import df_laureates, df_pop, df_prizes, lastyearincluded
from filters import count_per_country, replace_country_designations, standard_filter


def generate_choroplethglobe(data=df_laureates, country="birth", gender="all", categories="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generates a 3D globe with the number of Nobel Laureates per Country.

    Parameters:
    - data (pd.DataFrame): The input DataFrame. Default: df_laureates.
    - country (str): The column name representing the country. Default: "birth".
        - "birth"
        - "affiliation"
        - "death"
    - category (str): The Nobel Prize category to filter by. Allowed values:
        - "all" (no filtering)
        - "Physics"
        - "Chemistry"
        - "Physiology or Medicine"
        - "Literature"
        - "Peace"
        - "Economic Sciences"
        Default: "all".
    - gender (str): The gender to filter by. Allowed values:
        - "all" (no filtering)
        - "male"
        - "female"
        Default: "all".

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The Plotly figure object representing the globe.
    """

    # Filter
    df_filtered = standard_filter(data, categories, gender, timerange, timerange_field)

    # Count
    country = replace_country_designations(country)
    data = count_per_country(df_filtered, country)

    # Create figure
    fig = go.Figure(data=go.Choropleth(
        locations=data['ISO3'],
        z=data['Count'],
        colorscale=cf.c_colorscale_teal,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar=dict(
            title='No of Laureates',
            len=0.9,  # Adjust the height
            x=1,  # Adjust the x position (default is 1, which is far right)
            xanchor='left',  # Align colorbar with its left side at x=0.9
            y=1,  # Center the colorbar vertically
            yanchor='top'  # Align colorbar around its middle at y=0.5
        ),
    )) 

    fig.update_layout(      
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r":0,"t":0,"l":0,"b":0},
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 11,
        ),
        # showlegend = True,
        # title=dict(
        #     text = "1.b: Nobel Prizes per Country (Country of Birth)",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        # width=1200, 
        height=800,
        autosize=True,
        geo=dict(
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
        )
    )

    # Hover label template
    fig.update_traces(
        customdata=data[['Country']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +  # Show Country name (specified in customdata)
            "%{z}" + # Show the count
            "<extra></extra>"  # Hide the trace
        )
    )

    # Hover label styling comes from the theme template (theme-aware surface color).

    return fig


def generate_scattermapbox_cities(data=df_laureates, city="birth", gender="all", categories="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generates a Scatter Mapbox with the number of Nobel Laureates per City.

    Parameters:
    - data (pd.DataFrame): The input DataFrame. Default: df_laureates.
    - City (str): The column name representing the country. Default: "birth".
        - "birth"
        - "affiliation"
        - "death"
    - categories (str): The Nobel Prize categories to filter by. Allowed values:
        - "all" (no filtering)
        - "Physics"
        - "Chemistry"
        - "Medicine"
        - "Literature"
        - "Peace"
        - "Economic Sciences"
        Default: "all".
    - gender (str): The gender to filter by. Allowed values:
        - "all" (no filtering)
        - "male"
        - "female"
        Default: "all".

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The Plotly figure object representing the globe.
    """
    # Filter
    data = standard_filter(data, categories, gender, timerange, timerange_field)

    # Function to add small jitter to coordinates to avoid overlap
    def add_jitter(coordinates, scale=0.05):
        return coordinates + np.random.uniform(-scale, scale, size=len(coordinates))

    # Filter city type
    if city == "birth":
        # Add jitter to latitude and longitude
        latitudes = add_jitter(data['BirthCityNowLat'])
        longitudes = add_jitter(data['BirthCityNowLon'])
        hover_text = [
            f"Name: {name}<br>City: {city}<br>Country: {country}<br>Date: {date}"
            for name, city, country, date in zip(data['AwardeeDisplayName'], data['BirthCity'], data['BirthCountry'], data['BirthDate'])
        ]
    elif city == 'affiliation':

          # Filter out rows where death city coordinates are missing
        data_filtered = data.dropna(subset=['Prize0_Affiliation0_CityLatitude', 'Prize0_Affiliation0_CityLongitude'])
        
        if data_filtered.empty:
            return go.Figure()  # Return an empty figure if no valid data is available

        # Use death city coordinates and data
        latitudes = add_jitter(data['Prize0_Affiliation0_CityLatitude'])
        longitudes = add_jitter(data['Prize0_Affiliation0_CityLongitude'])
        hover_text = [
            f"Name: {name}<br>City: {city}<br>Country: {country}<br>Prize Award Date: {date}"
            for name, city, country, date in zip(data['AwardeeDisplayName'], data['Prize0_Affiliation0_City'], data['Prize0_Affiliation0_Country'], data['Prize0_DateAwarded'])
        ]
    elif city == 'death':

          # Filter out rows where death city coordinates are missing
        data_filtered = data.dropna(subset=['DeathCityLat', 'DeathCityLon'])
        
        if data_filtered.empty:
            return go.Figure()  # Return an empty figure if no valid data is available

        # Use death city coordinates and data
        latitudes = add_jitter(data['DeathCityLat'])
        longitudes = add_jitter(data['DeathCityLon'])
        hover_text = [
            f"Name: {name}<br>City: {city}<br>Country: {country}<br>Date: {date}"
            for name, city, country, date in zip(data['AwardeeDisplayName'], data['DeathCity'], data['DeathCountryNow'], data['DeathDate'])
        ]

    # Create the map using CartoDB Positron
    fig = go.Figure(go.Scattermap(
        lat=latitudes,    # Jittered Latitude coordinates
        lon=longitudes,    # Jittered Longitude coordinates
        mode='markers',
        marker=go.scattermap.Marker(
            size=9,   # Marker size
            color=th.SPECTRUM_LIGHT["Physics"],  # spectrum blue; swaps to Neon in dark mode
            opacity=0.8
        ),
        text=data['AwardeeDisplayName'],  # Laureate name (used for hover)
        hoverinfo='text',  # Tooltip content
        hovertext=hover_text  # Use dynamically generated hover text
    ))

    # Update layout of the map
    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r":0,"t":0,"l":0,"b":0},
        # font + hoverlabel styling come from the theme template (theme-aware).
        height = 800,
        # go.Scattermap (MapLibre) reads layout.map — the earlier layout.mapbox
        # settings were silently ignored. Style swaps to carto-darkmatter in dark mode.
        map=dict(
            style=th.MAPBOX_STYLE_LIGHT,
            zoom=1,  # Set default zoom level
            center=dict(lat=20, lon=0)  # Default map center
        ),
    )

    return fig


def prepare_data_prizespercountry(data, country="birth"):

    """
    Calculates prizes per country.

    Args:
        data (pd.DataFrame):
            The raw DataFrame containing Nobel Laureate data with columns like 'BirthCountryNow', 
            'Prize0_AwardYear', 'DeathCountryNow', etc.
        country (str):
            A string indicating which country field to use: 
            'birth' uses BirthCountryNow, 
            'affiliation' uses Prize0_Affiliation0_CountryNow, 
            'death' uses DeathCountryNow. 
            Any other value defaults to using BirthCountryNow.
    
    Returns:
        tuple:
            - **df_nlpc_complete (pd.DataFrame)**: 
              Dataframe with prizes per country.
    """
        

    # Get relevant columns from the master table, based on user preference
    if country=="birth":
        df_nlpc = data[['BirthCountryNow', 'LaureateGender', 'Prize0_AwardYear']]
    elif country=="affiliation":
        df_nlpc = data[['Prize0_Affiliation0_CountryNow', 'LaureateGender', 'Prize0_AwardYear']]
    elif country=="death":
        #df_nlpc = data[['DeathCountryNow', 'LaureateGender', 'Prize0_AwardYear']]
        df_nlpc = data[['DeathCountryNow', 'LaureateGender', 'DeathDate']]
        df_nlpc = df_nlpc.dropna() # drop nan
        df_nlpc['DeathDate'] = pd.to_datetime(df_nlpc['DeathDate'], errors='coerce').dt.year.astype(int) # convert dates to years (as int)
        df_nlpc = df_nlpc[(df_nlpc[['DeathCountryNow', 'LaureateGender', 'DeathDate']] != 0).all(axis=1)] # drop all rows where there are 0s somewhere
    else:
        df_nlpc = data[['BirthCountryNow', 'LaureateGender', 'Prize0_AwardYear']]

    


    # rename for simplicity
    df_nlpc.columns = ['Country', 'Gender', 'Year']

    # Merge the counts list and the ISO3 list
    # df_nlpc = pd.merge(df_nlpc, df_iso, on="Country")

    # Calculate the number of prizes per country per year
    df_nlpc_count = df_nlpc.groupby(['Year', 'Country']).size().reset_index(name='Prizes')

    # Sort the data by 'Country' and 'Year'
    df_nlpc_count = df_nlpc_count.sort_values(by=['Country', 'Year'])

    # Calculate the running sum (cumulative sum) per country
    df_nlpc_count['RunningSum'] = df_nlpc_count.groupby('Country')['Prizes'].cumsum()

    # Create a complete index of years for each country
    all_years = pd.DataFrame({'Year': range(df_nlpc_count['Year'].min(), df_nlpc_count['Year'].max() + 1)})
    all_countries = df_nlpc_count['Country'].unique()
    complete_index = pd.MultiIndex.from_product([all_years['Year'], all_countries], names=['Year', 'Country'])

    # Reindex the DataFrame to include all years for each country
    df_nlpc_complete = df_nlpc_count.set_index(['Year', 'Country']).reindex(complete_index).reset_index()

    # Forward fill the missing values for the running sum
    df_nlpc_complete['RunningSum'] = df_nlpc_complete.groupby('Country')['RunningSum'].ffill().fillna(0)
    df_nlpc_complete['Prizes'] = df_nlpc_complete['Prizes'].fillna(0)

    #df_nlpc.columns =["Year", "Country", "PrizeInThisYear", "Prizes"]
    df_nlpc_complete = df_nlpc_complete.sort_values(by=['Year', 'Country'])

    return df_nlpc_complete


def prepare_data_bubbles_population(df_nlpc_complete, df_pop=df_pop, interval=5):
    """
    Merge Nobel Laureate prize data with population data for bubble charts.
    
    Prepares data for animated population-based bubble charts by merging
    cumulative prize counts with population data and computing necessary
    transformations (log scales, normalization).
    
    Args:
        df_nlpc_complete (pd.DataFrame): Prize count data from prepare_data_prizespercountry().
        df_pop (pd.DataFrame): Population data by country and year.
            Uses K/M/B notation (e.g., '1.5M' for 1.5 million).
            Defaults to df_pop (gapminder.org data).
        interval (int): Year interval for animation frames.
            Higher values = faster loading, fewer frames.
            Defaults to 5 (shows every 5th year).
            
    Returns:
        tuple: (df_result, y_range_max, bubblesize)
            - df_result (pd.DataFrame): Processed data with log-transformed values.
            - y_range_max (float): Y-axis upper limit for plotting.
            - bubblesize (np.ndarray): Scaled bubble sizes.
            
    Performance Notes:
        - Interval=5 recommended for balance of detail vs. performance.
        - Lower intervals significantly increase figure size and load time.
    """
    # Work on copies to avoid side effects
    df_nlpc = df_nlpc_complete.copy()
    df_pop_work = df_pop.copy()
     
    # Add a small constant to ensure minimum bubble size
    df_nlpc['AdjustedSize'] = df_nlpc['RunningSum'] + 10

    # OPTIMIZATION: Use vectorized string formatting instead of apply+lambda
    df_nlpc['Text'] = df_nlpc['Country'] + ': ' + df_nlpc['RunningSum'].astype(int).astype(str)

    # Clean POP data
    if "ISO3" in df_pop_work.columns:
        df_pop_work = df_pop_work.drop(columns=["ISO3"])

    # OPTIMIZATION: Vectorized population conversion using regex
    def convert_population_vectorized(col):
        """Convert population strings (e.g., '1.5M', '300k') to numeric values."""
        if col.dtype == object:
            # Handle billions
            mask_b = col.str.contains('B', na=False)
            # Handle millions
            mask_m = col.str.contains('M', na=False)
            # Handle thousands
            mask_k = col.str.contains('k', na=False)
            
            result = pd.to_numeric(col.str.replace('[BMk]', '', regex=True), errors='coerce')
            result = result.where(~mask_b, result * 1_000_000_000)
            result = result.where(~mask_m, result * 1_000_000)
            result = result.where(~mask_k, result * 1_000)
            return result
        return col

    # Apply vectorized conversion to all year columns
    year_columns = [c for c in df_pop_work.columns if c != 'country']
    for column in year_columns:
        df_pop_work[column] = convert_population_vectorized(df_pop_work[column])

    # Melt the population DataFrame
    df_pop_melted = df_pop_work.melt(id_vars=["country"], var_name="year", value_name="population")
    df_pop_melted.columns = ["Country", "Year", "Population"]
    
    # OPTIMIZATION: Convert Year to int once, before merge
    df_pop_melted['Year'] = pd.to_numeric(df_pop_melted['Year'], errors='coerce').astype('Int64')

    # Merge dataframes
    df_merged = df_nlpc.merge(df_pop_melted, how='left', on=["Country", "Year"])

    # OPTIMIZATION: Compute all derived columns in one pass using assign()
    df_merged = df_merged.assign(
        Prizes=pd.to_numeric(df_merged['RunningSum'], errors='coerce'),
        Population=pd.to_numeric(df_merged['Population'], errors='coerce')
    )
    
    # Compute ratios
    df_merged['PrizesPerPop'] = df_merged['RunningSum'] / df_merged['Population']
    df_merged['PrizesPer1MPop'] = df_merged['PrizesPerPop'] * 1_000_000

    # Log transforms
    df_merged['LogPopulation'] = np.log(df_merged['Population'] + 0.01)
    df_merged['LogPrizesPer1MPop'] = np.log(df_merged['PrizesPer1MPop'] + 1)

    # OPTIMIZATION: Filter to interval years early to reduce data size
    mask = (df_merged["Year"] % interval == 0) | (df_merged["Year"] == lastyearincluded)
    df_result = df_merged.loc[mask].copy()
    
    # Remove rows with NaN values in critical columns (countries without population data)
    df_result = df_result.dropna(subset=['LogPopulation', 'LogPrizesPer1MPop'])
    
    # Round values
    df_result['LogPopulation'] = df_result['LogPopulation'].round(4)
    df_result['LogPrizesPer1MPop'] = df_result['LogPrizesPer1MPop'].round(2)

    # Compute normalized population and bubble sizes
    pop_min = df_result['Population'].min()
    pop_max = df_result['Population'].max()
    df_result['PopulationNormalized'] = ((df_result['Population'] - pop_min) / (pop_max - pop_min)).fillna(0)
    
    y_range_max = df_result['LogPrizesPer1MPop'].max() * 1.1
    bubblesize = np.maximum(df_result['PopulationNormalized'] * 500, 1)

    return df_result, y_range_max, bubblesize


def generate_bubbles_perpopulation(data=df_prizes, country="birth", gender="all", categories="all", df_pop=df_pop, interval=5, timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generate an animated population-based bubble chart of Nobel Prize data.
    
    Creates an interactive scatter plot showing Nobel Prize distribution
    relative to country population over time, with animation frames for
    each time interval.
    
    Args:
        data (pd.DataFrame): Nobel Prize data. Defaults to df_prizes.
        country (str): Country field to use - 'birth', 'affiliation', or 'death'.
            Defaults to 'birth'.
        gender (str): Gender filter - 'all', 'male', or 'female'. Defaults to 'all'.
        categories (str | list): Category filter. Defaults to 'all'.
        df_pop (pd.DataFrame): Population data by country/year. Defaults to df_pop.
        interval (int): Year interval between animation frames (higher = faster).
            Defaults to 5.
        timerange (list[int]): [start_year, end_year]. Defaults to [1901, lastyearincluded].
        timerange_field (str): Date field for filtering. Defaults to 'award'.
        
    Returns:
        plotly.graph_objs.Figure: Interactive animated bubble chart.
        
    Performance Notes:
        - Uses pre-indexed data lookups for hover updates (O(1) vs O(n))
        - Interval parameter significantly affects generation time
        - Consider interval=10 for faster initial loads
    """

    data = standard_filter(data, categories, gender, timerange, timerange_field)
    df_nlpc_complete = prepare_data_prizespercountry(data, country)
    plot_data, y_range_max, bubblesize = prepare_data_bubbles_population(df_nlpc_complete, df_pop, interval)

    fig = px.scatter(plot_data, 
                    x="LogPopulation", 
                    y="LogPrizesPer1MPop", 
                    size="LogPrizesPer1MPop", 
                    color="Country",
                    color_continuous_scale=cf.c_colorscale_palette,
                    hover_name="Country", 
                    animation_frame="Year", 
                    range_x=[np.log(50_000), np.log(2_000_000_000)],
                    range_y=[0, y_range_max],                                  
                    text="Text",
    )

    fig.update_layout(
            template='nbl_light',
            plot_bgcolor=cf.c_plot_background,
            margin={"r":0,"t":30,"l":0,"b":60},
            font=dict(
                family = 'IBM Plex Sans, sans-serif',
                size = 11,
            ),
            showlegend = False,
            autosize=True,
            height=800,
    )

    # OPTIMIZATION: Pre-build lookup dictionaries for O(1) access instead of O(n) filtering
    # Group data by country for initial frame
    country_groups = {name: group for name, group in plot_data.groupby('Country')}
    
    # Build (country, year) -> row lookup for animation frames
    hover_data_lookup = {}
    for _, row in plot_data.iterrows():
        key = (row['Country'], row['Year'])
        hover_data_lookup[key] = [row['Country'], row['Prizes'], row['Population'], row['PrizesPer1MPop']]

    # Common hovertemplate
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Prizes: %{customdata[1]}<br>"
        "Population: %{customdata[2]:,}<br>"
        "Prizes per 1M inhabitants: %{customdata[3]:.1f}"
        "<extra></extra>"
    )

    # Update initial traces using pre-grouped data
    for trace in fig.data:
        country_name = trace.name
        if country_name in country_groups:
            country_data = country_groups[country_name]
            trace.customdata = country_data[['Country', 'Prizes', 'Population', 'PrizesPer1MPop']].values
            trace.hovertemplate = hovertemplate

    # Update animation frames using lookup dictionary
    for frame in fig.frames:
        frame_year = int(frame.name)
        for trace in frame.data:
            country_name = trace.name
            key = (country_name, frame_year)
            if key in hover_data_lookup:
                trace.customdata = [hover_data_lookup[key]]
                trace.hovertemplate = hovertemplate

    # Hover label styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )

    # Define tick values and labels for axes
    tickvals_x = np.log([50_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_500_000_000, 2_000_000_000])
    ticktext_x = ['50k', '100k', '1M', '10M', '100M', '1.5B', '2.0B']
    tickvals_y = np.power([0, 1, 2, 3, 4, 5, 10, 20, 50], (1/4))
    ticktext_y = ['0', '1', '2', '3', '4', '5', '10', '20', '50']

    fig.update_xaxes(tickvals=tickvals_x, ticktext=ticktext_x, title="Population (log-scale)")
    fig.update_yaxes(tickvals=tickvals_y, ticktext=ticktext_y, title="Prizes Per Population (log-scale)")

    # Text label position
    fig.update_traces(textposition='top center')

    # Range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True, 
                range=[plot_data["LogPopulation"].min(), plot_data["LogPopulation"].max()],
            ),  
            type="linear",
        )
    )

    fig.update_xaxes(rangeslider_thickness = 0.05)

    # Animation slider
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.0,
            y=-0.3,
            xanchor="left",
            yanchor="bottom"
        )],
        sliders=[dict(y=-0.1)]
    )

    return fig


def generate_bar_percountry(data=df_prizes, country="birth", gender="all", categories="all", timerange=[1901, lastyearincluded], timerange_field="award", runningsum=False):
    """
    Generates a stacked bar chart of Nobel Prizes per country per year.
    
    This function:
      1. Applies a standard filter on the input data using `standard_filter()` to limit rows by categories and gender.
      2. Prepares the data by calling `prepare_data_stacked_bar()`
      3. Creates a Plotly bar chart. 
    
    Args:
        data (pd.DataFrame, optional):
            Nobel Prize data in a pandas DataFrame. Defaults to `df_prizes` (a global variable in this script). 
            Must contain columns appropriate for filtering by categories/gender and merging with population data.
        country (str, optional):
            Country type for merging Nobel data: 
              - "birth" (BirthCountryNow), 
              - "affiliation" (Prize0_Affiliation0_CountryNow), 
              - "death" (DeathCountryNow).
            Defaults to "birth".
        gender (str, optional):
            Filter the data for a specific gender ("male", "female", etc.). Using "all" applies no gender filter.
            Defaults to "all".
        categories (str, optional):
            Filter the data for a specific Nobel categories ("physics", "chemistry", etc.). Using "all" applies 
            no categories filter. Defaults to "all".
        runningsum (boolean):
            True: returns a plot with a running sum
            False: returns a plot with individual values per year (default)

    Returns:
        plotly.graph_objs._figure.Figure:
            A Plotly figure object representing the bar chart.
    
    Notes:
        - Internally calls `standard_filter()` to reduce the dataset based on categories/gender.
        - Returns a fully configured Plotly figure ready for interactive display or further styling.
    
    """

    # Filter
    data = standard_filter(data, categories, gender, timerange, timerange_field)

    # Prepare
    df_nlpc_complete = prepare_data_prizespercountry(data, country)

    # Return running sum (or not)
    if runningsum:
        data = df_nlpc_complete.pivot(index='Year', columns='Country', values='RunningSum').fillna(0)
    else:
        data = df_nlpc_complete.pivot(index='Year', columns='Country', values='Prizes').fillna(0)

    # Reset index to include 'Year' as a column
    data = data.reset_index()

    # Melt the data to include 'Country' as a row
    data_melted = data.melt(id_vars='Year', var_name='Country', value_name='Prizes')

    # Create the bar chart using melted data
    fig = px.bar(
        data_melted,
        x='Year',
        y='Prizes',
        color='Country',  # Color by country
        color_discrete_sequence=cf.c_colorscale_palette_light,
        labels={'Prizes': 'Number of Prizes', 'Country': 'Country'}
    )

    # Assign `customdata` for each trace separately
    for trace in fig.data:
        country_name = trace.name  # Get the country name for this trace
        country_data = data_melted[data_melted['Country'] == country_name]  # Filter for this country

        trace.customdata = country_data[['Year', 'Prizes']].values  # Assign Year and Prizes to customdata
        trace.hovertemplate = (
            "<b>" + country_name + "</b><br>" +  # Country name (manually added)
            "Year: %{customdata[0]}<br>" +  # Year
            "Prizes: %{customdata[1]}" +  # Number of prizes
            "<extra></extra>"
        )

    # Update layout and hover styling
    fig.update_layout(
        barmode='stack',
        xaxis_title='Year',
        yaxis_title='Number of Prizes',
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        font=dict(
            family='IBM Plex Sans, sans-serif',
            size=11,
        ),
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )


    return fig
