"""
Demography figures: 3D surface, donuts, sunburst, first names.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from data import df_laureates, df_prizes, lastyearincluded
from filters import standard_filter


def prepare_data_3dsurface(data, categories, gender, timerange, timerange_field):
    """
    Prepares data for a 3D surface plot.
    
    This function:
      1. Collects the necessary data (from df_prizes).
      2. Calculates decades.
      3. Counts prizes for men and women
      4. Pivots the data
      5. Returns two df.
    
    Args:
        data (pd.DataFrame):
            Nobel Prize data in a pandas DataFrame.

    Returns:
        df_prizes_women, df_prizes_men (Pandas Dataframe)
            Two dfs allowing to show two surfaces in the plot.    
    """

    data = standard_filter(data, categories, gender, timerange, timerange_field)


    # To ensure that we don't accidentally modify the passed original dataframe.
    data = data.copy()

    # Replace catgeory names with abbreviations so they fit better in the plot
    data.loc[:,"Prize0_Category"] = data["Prize0_Category"].replace({"Medicine": "Med", "Physics": "Phys", "Chemistry": "Chem", "Economic Sciences": "Eco", "Literature": "Lit", "Peace": "Pea"})


    # Get required data from main df
    df_prizes_pergender_perdecade_a = data[['LaureateGender', 'Prize0_AwardYear', 'Prize0_Category']].copy()

    # Calculate the decade and drop the year
    df_prizes_pergender_perdecade_a['Decade'] = (df_prizes_pergender_perdecade_a['Prize0_AwardYear'] // 10) * 10  # The //-operator rounds down.
    df_prizes_pergender_perdecade_a.drop(columns=['Prize0_AwardYear'], inplace=True) 

    # Count the genders per decade per discipline (using the groupby-function)
    df_prizes_pergender_perdecade_b = df_prizes_pergender_perdecade_a.groupby(['Decade', 'LaureateGender', 'Prize0_Category']).size().reset_index(name='Count')

    # Remove rows with gender "male" or "org" for women
    df_prizes_women = df_prizes_pergender_perdecade_b[(df_prizes_pergender_perdecade_b['LaureateGender'] != 'male') & (df_prizes_pergender_perdecade_b['LaureateGender'] != 'org')]

    # Remove rows with gender "female" or "org" for men
    df_prizes_men = df_prizes_pergender_perdecade_b[(df_prizes_pergender_perdecade_b['LaureateGender'] != 'female') & (df_prizes_pergender_perdecade_b['LaureateGender'] != 'org')]

    # 2: As we have only women left, delete the column gender
    df_prizes_women = df_prizes_women.drop(columns=['LaureateGender'])
    df_prizes_men = df_prizes_men.drop(columns=['LaureateGender'])

    # 3: Pivot the table, so that decades apear on the x-axis, and disciplines on the y-axis. Counts are now values, with zeroes filled in for empty values.
    df_prizes_women = df_prizes_women.pivot_table(index='Prize0_Category', columns='Decade', values='Count', fill_value=0)
    df_prizes_men = df_prizes_men.pivot_table(index='Prize0_Category', columns='Decade', values='Count', fill_value=0)

    return df_prizes_women, df_prizes_men


def generate_3dsurface_pergender(data=df_prizes, categories="all", gender="female", timerange=[1901, lastyearincluded], timerange_field="award", height=800):
    """
    Generates 3D surface plot, showing prizes to men and women.
    
    This function:
      1. Prepares the data by calling `prepare_data_3dsurface()`
      2. Defines the surfaces to be shown based on the input in 'gender'
      3. Creates a Plotly bar chart. 
    
    Args:
        data (pd.DataFrame, optional):
            Nobel Prize data in a pandas DataFrame. Defaults to `df_prizes`
        gender (str, optional):
            Filter the data for a specific gender ("male", "female", "all".). Using "all" shows two surfaces for men and women.
            Defaults to "all".

    Returns:
        plotly.graph_objs._figure.Figure:
            A Plotly figure object representing the bar chart.
    
    Notes:
        - Internally calls `prepare_data_3dsurface()` to prepare the dataset.
        - Returns a fully configured Plotly figure ready for interactive display or further styling.
    
    """
    # Prepare data
    data_w, data_m = prepare_data_3dsurface(data, categories, gender, timerange, timerange_field)

    # Define axes dynamically
    x = list(data_w.columns)  # Decades from the pivoted DataFrame columns
    y = list(data_w.index)    # Categories from the pivoted DataFrame index

    # Filter gender and determine surfaces
    if gender == "all":
        data = [
            go.Surface(z=data_w.values, y=y, x=x, colorscale=cf.c_colorscale_palette, showscale=False, opacity=0.7),
            go.Surface(z=data_m.values, y=y, x=x, colorscale=cf.c_colorscale_palette, showscale=False, opacity=0.7)
        ]
    elif gender == "female":
        data = [
            go.Surface(z=data_w.values, y=y, x=x, colorscale=cf.c_colorscale_palette, showscale=False, opacity=0.7)
        ]
    elif gender == "male":
        data = [
            go.Surface(z=data_m.values, y=y, x=x, colorscale=cf.c_colorscale_palette, showscale=False, opacity=0.7)
        ]

    # Initiate figure
    fig = go.Figure(data=data)

    # Add layout details
    fig.update_layout(
        autosize=True,
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        scene=dict(
            xaxis=dict(tickvals=x, ticktext=x, title='Decades', autorange='reversed'),
            yaxis=dict(title='Categories'),
            zaxis=dict(title='Prizes'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=-0.3),
                up=dict(x=0, y=0, z=2)
            ),
        ),
        # width = width,
        height = height,
        margin=dict(l=10, r=10, t=10, b=10),

        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    
    )

    # Hover label template
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>" +  # Category
            "Decade: %{x}+<br>" + # Decade
            "Prizes: %{z}" # Prize count
            "<extra></extra>"  # Hide the trace
        )
    )

    # Hover label styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )
    return fig


def prepare_data_donuts(data, categories, gender, timerange, timerange_field, characteristic):
    """
    Prepares data for a donut chart by filtering and aggregating counts 
    based on the specified characteristic.

    Args:
        data (DataFrame): The input dataset containing laureate information.
        categories (str): The categories filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): The gender filter (e.g., 'male', 'female', or 'all').
        characteristic (str): The characteristic to aggregate by 
            ('gender', 'ethnicity', or 'religion').

    Returns:
        DataFrame: A DataFrame with columns:
            - 'Label': The unique values of the specified characteristic.
            - 'Count': The count of occurrences for each label.
    """

    # Standard filter
    data = standard_filter(data, categories, gender, timerange, timerange_field, characteristic)

    if characteristic == "gender":
        # Data for gender
        df_gender_counts = data['LaureateGender'].value_counts().reset_index()
        df_gender_counts.columns = ['Label', 'Count']
        return df_gender_counts

    elif characteristic == "ethnicity":
        # Data for ethnicity
        df_ethnicity_counts = data['Ethnicity'].value_counts().reset_index()
        df_ethnicity_counts.columns = ['Label', 'Count']
        return df_ethnicity_counts

    elif characteristic == "religion":
        # Data for religion
        df_religion_counts = data['Religion'].value_counts().reset_index()
        df_religion_counts.columns = ['Label', 'Count']
        return df_religion_counts

    else:
        # Default empty DataFrame
        df_empty = pd.DataFrame([["No Data", 1]], columns=['Label', 'Count'])
        return df_empty


def generate_donut(data=df_laureates, categories="all", gender="all", characteristic="gender", timerange=[1901, lastyearincluded], timerange_field="award", dark=False):
    """
    Generates a donut chart visualizing the distribution of a specified 
    characteristic (e.g., gender, ethnicity, or religion) among Nobel laureates.

    Args:
        data (DataFrame): The input dataset containing laureate information (default: df_laureates).
        categories (str): The categories filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): The gender filter (e.g., 'male', 'female', or 'all').
        characteristic (str): The characteristic to visualize 
            ('gender', 'ethnicity', 'religion').

    Returns:
        plotly.graph_objects.Figure: A Plotly donut chart displaying the distribution 
        of the specified characteristic.
    """

    # Prepare data
    data = prepare_data_donuts(data, categories, gender, timerange, timerange_field, characteristic)

    if data.empty:
        print("Error: No data available for generating the plot.")
        fig = {"data": [], "layout": {"title": "Error generating plot"}}
    else:

        # Extract labels and values
        labels = data['Label'].tolist()
        values = data['Count'].tolist()
        characteristic_name = characteristic.capitalize()

        # Create the donut chart (Nobel-Spektrum styling, theme-aware)
        n = th.neutrals(dark)
        # In dark mode lead with the Neon colorway, otherwise Jewel (extended for many slices).
        donut_palette = (th.colorway(True) + th.colorway(False)) if dark else (th.colorway(False) + th.colorway(True))
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.64,
                    sort=False,
                    direction='clockwise',
                    marker=dict(
                        colors=donut_palette,
                        line=dict(color=n["surface"], width=3)  # slice separators in the surface color
                    ),
                    showlegend=False,
                    textinfo='label+percent',
                    textposition='outside',
                    textfont=dict(family=th.FONT_BODY, size=12, color=n["text_muted"]),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            template=th.get_template(dark),
            plot_bgcolor=cf.c_plot_background,
            annotations=[
                dict(
                    text=characteristic_name,
                    x=0.5,
                    y=0.5,
                    font=dict(family=th.FONT_DISPLAY, size=16, color=n["text_primary"]),
                    showarrow=False,
                    xanchor="center"
                )
            ],
            autosize=True,
            margin=dict(l=24, r=24, t=24, b=24),

            hoverlabel=dict(
                bgcolor=n["surface"],
                font_size=12,
                font_family=th.FONT_BODY
            )
        )

    return fig


def prepare_data_sunburst(data, year, path, categories, gender, timerange, timerange_field):
    """
    Prepares data for a sunburst chart by grouping and counting based on the specified path.

    Args:
        data (DataFrame): The input dataset containing laureate information.
        year (str or int): The year to filter the data ('last' for the latest year or a specific year).
        path (list): A list of columns representing the hierarchical path for the sunburst chart.

    Returns:
        DataFrame: A DataFrame containing the grouped counts with the specified path columns 
        and a 'count' column representing the number of occurrences.
    """
   
    data = standard_filter(data, categories, gender, timerange, timerange_field)

    if year=="last":
        data = data[data['Prize0_AwardYear'] == lastyearincluded]
    else:
        data = data

    # Fill empty columns, so they are not excluded for the count    
    data['BirthCountryNow'] = data['BirthCountryNow'].fillna('Unknown')
    data['Prize0_Category'] = data['Prize0_Category'].fillna('Unknown')
    data['LaureateGender'] = data['LaureateGender'].fillna('Unknown')

    # Group by Category and Gender to get the counts
    data_count = data.groupby(path).size().reset_index(name='count')
    data_count.replace({'Economic Sciences': 'Economics', 'United States': 'USA'}, inplace=True)

    return data_count


def generate_sunburst(data=df_laureates, year="all", path=['Prize0_Category', 'LaureateGender', 'BirthCountryNow'], categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", dark=False):
    """
    Generates a sunburst chart visualizing the distribution of Nobel laureates 
    based on the specified hierarchical path.

    Args:
        data (DataFrame): The input dataset containing laureate information (default: df_laureates).
        year (str or int): The year to filter the data ('all' for all years or 'last' for the latest year).
        path (list): A list of columns representing the hierarchical path for the sunburst chart.

    Returns:
        plotly.graph_objects.Figure: A Plotly sunburst chart showing the hierarchical distribution 
        with customized colors based on prize categories.
    """

    data = prepare_data_sunburst(data, year, path, categories, gender, timerange, timerange_field)

    if data.empty:
        fig = {"data": [], "layout": {"title": "Error generating plot"}}
    else:
        n = th.neutrals(dark)
        spec = th.spectrum(dark)
        fig = px.sunburst(
            data,
            path=path,
            values='count',
            color='Prize0_Category',
            color_discrete_map={
                "Medicine": spec["Medicine"],
                "Physics": spec["Physics"],
                "Chemistry": spec["Chemistry"],
                "Literature": spec["Literature"],
                "Peace": spec["Peace"],
                "Economics": spec["Economics"],
            }
        )

        fig.update_layout(
            template=th.get_template(dark),
            plot_bgcolor=cf.c_plot_background,
            margin=dict(t=6, b=6, l=6, r=6),
            hoverlabel=dict(
                bgcolor=n["surface"],
                font_size=12,
                font_family=th.FONT_BODY
            )
        )

            # Hover label template + Spektrum segment styling
        fig.update_traces(
            branchvalues='total',
            insidetextfont=dict(family=th.FONT_BODY, color="#FFFFFF"),
            marker=dict(line=dict(color=n["surface"], width=2)),
            hovertemplate=(
                "Path: %{id}<br>"  # Show the full hierarchical path
                "Count: %{value}"  # Show the count
            )
        )


    return fig


def generate_mostcommon_firstnames(data=df_laureates, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generates a bar chart showing the most common first names among Nobel laureates.

    
    """
    data = standard_filter(data, categories, gender, timerange, timerange_field)

    data = data["LaureateNameFirst"].str.split(' ').str[0]
    df_data = data.to_frame()

    df_data = df_data.dropna()  # Option 1: Drop rows with NaN

    df_filtered = df_data[
        ~df_data['LaureateNameFirst'].str.match(r'^[A-Z]\.$') &  # Drop initials like "J."
        ~df_data['LaureateNameFirst'].str.contains(r'^Sir$', case=False)  # Drop "Sir" (case insensitive)
    ]

    top_names = df_filtered['LaureateNameFirst'].value_counts().head(25)

    # Cycle the Nobel-Spektrum colorway (swaps to Neon in dark mode) instead of the
    # pre-redesign palette.
    spectrum = th.colorway(False)
    bar_colors = [spectrum[i % len(spectrum)] for i in range(len(top_names))]

    # Create a bar chart (single trace so colors/hover data stay aligned per bar)
    fig = px.bar(
        x=top_names.values,
        y=top_names.index,
        orientation='h',  # Horizontal bar chart
        labels={'x': 'Count', 'y': 'Name'},
    )

    fig.update_layout(
            xaxis_title="Occurences",
            yaxis_title="Most Common First Names",
            template='nbl_light',
            plot_bgcolor=cf.c_plot_background,
            #yaxis=dict(ticksuffix="   "),
            yaxis=dict(
                tickmode="linear",  # Ensures all labels are shown
                tickfont=dict(size=10),  # Adjust font size for better readability
                ticksuffix="   ",  # Add spaces to the end of each label for better alignment
                automargin=True,
            ),
            xaxis=dict(
                automargin=True,
            ),
            margin={"r":20,"t":20,"l":0,"b":0},

            # font color comes from the theme template (theme-aware)
            font=dict(
                family = 'IBM Plex Sans, sans-serif',
                size = 11,
            ),

            showlegend = False,

            # title=dict(
            #     text = "Most common First Names",
            #     font=dict(size = 20),
            #     x = 0,                            # Left align the title
            #     xanchor = 'left',                 # Align to the left edge
            #     y = 0.97,                         # Adjust Y to position title above the map
            #     yanchor = 'top',                  # Anchor at the top of the title box
            # ),

            autosize=True
        )

    customdata = top_names.reset_index().values  # Prepare customdata for hovertemplate, one row per bar

    fig.update_traces(
        marker_color=bar_colors,
        customdata=customdata,
        hovertemplate=(
            "<b>Name:</b> %{customdata[0]}<br>" +
            "Count: %{customdata[1]}<br>" +
        "<extra></extra>"  # Hide the trace info
        )
    )
    
    

    # Hover label styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )

    return fig
