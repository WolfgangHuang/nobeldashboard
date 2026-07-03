"""
Time figures: publication-prize time gap, age at award.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from sklearn.linear_model import LinearRegression
from data import df_laureates, df_lifeexpectancy, df_prizes, lastyearincluded
from filters import standard_filter


def prepare_data_histogram_timegap(data, categories, gender, timerange, timerange_field, datasource):
    """
    Filters and prepares data for a timegap histogram.

    Args:
        data (DataFrame): The input data containing a 'PublicationSource' column. Usually df_laureates passed from the calling function.
        categories (str): The categories filter to apply.
        gender (str): The gender filter to apply.
        datasource (str): The data source to filter by 
            ('paper', 'chatgpt', or other).

    Returns:
        DataFrame: The filtered DataFrame with rows where 'PublicationSource' 
        is not empty and matches the specified filters.
    """

    data = standard_filter(data, categories, gender, timerange, timerange_field)

    # delete rows where there is no info on the timegap
    df_filtered = data[data['PublicationSource'].notna() & (data['PublicationSource'].str.strip() != "")]

    if datasource == "paper":
        df_filtered = df_filtered[(df_filtered['PublicationSource']) == "Harvard Dataverse"]

    elif datasource == "chatgpt":
        df_filtered = df_filtered[(df_filtered['PublicationSource']) == "ChatGPT 4c (September 2024)"]
    else:
        df_filtered = df_filtered
    
    return df_filtered


def generate_histogram_timegap(data=df_prizes, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", datasource="all"):
    """
    Generates a histogram visualizing the time gap between seminal papers 
    and Nobel Prizes.

    Args:
        data (DataFrame): The input dataset (default: df_prizes).
        categories (str): categories filter (e.g., 'Physics', 'Chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        datasource (str): Data source filter (e.g., 'paper', 'chatgpt', or 'all').

    Returns:
        plotly.graph_objects.Figure: A histogram of time gaps, colored by 
        prize categories, with custom styling and layout.
    """    

    data = prepare_data_histogram_timegap(data, categories, gender, timerange, timerange_field, datasource)

    fig = px.histogram(
        data,
        x="PublicationTimegap",
        color="Prize0_Category", 
        opacity=1,
        color_discrete_map={
            "Physics": cf.c_physics, 
            "Chemistry":cf.c_chemistry, 
            "Medicine": cf.c_medicine
        },
        nbins=100,  # Set number of bins to 1 per gap,
        #title="Histogram of Timegap Between Seminal Paper and Nobel Prize",
        labels={
            "Timegap": "Time Gap (years)",   # X-axis label
            "Count": "Number of Prizes"   # Y-axis label
        },
            barmode="group"
    )
    
    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r":0,"t":50,"l":0,"b":0},
        xaxis_title="Average Time Gap (years)",
        yaxis_title="Count",
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
        ),
        # title=dict(
        #     text = "3.a: Histogram of Timegap between Seminal Paper and Nobel Prize",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        legend_title_text="Prize Category",
        # width = 1000,
        # height = 600,
        autosize=True
        ),
    
    # Hover label template
    fig.update_traces(
        customdata=data[['Prize0_Category']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +  # Category
            "Timegap: %{x} years<br>" + # Timegap
            "Count: %{y}" + # Count
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


def generate_scatterbox_timegaptrend(data=df_laureates, df_lifeexpectancy=df_lifeexpectancy, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", datasource="both"):
    """
    Generates a trend visualization of the time gap between seminal papers 
    and Nobel Prizes, including trendlines and life expectancy comparisons.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        df_lifeexpectancy (DataFrame): Life expectancy data (default: df_lifeexpectancy).
        categories (str): categories filter (e.g., 'Physics', 'Chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').

    Returns:
        plotly.graph_objects.Figure: A line and scatter plot with trendlines, 
        categorized by prize, and overlaid with life expectancy trends.
    """

    data = prepare_data_histogram_timegap(data, categories, gender, timerange, timerange_field, datasource)
    
    fig = go.Figure()

    categories = data['Prize0_Category'].unique()

    colors = {
        "Physics": cf.c_physics, 
        "Chemistry":cf.c_chemistry, 
        "Medicine": cf.c_medicine
    }

    # Adding scatter points for each category
    for category in categories:
        subset = data[data['Prize0_Category'] == category]
        
        # Scatter points for each category
        scatter_trace = go.Scatter(
            x=subset["Prize0_AwardYear"], 
            y=subset["PublicationTimegap"],  
            mode='markers',  
            name=category,
            marker=dict(color=colors[category], size=8),
            customdata=subset[['Prize0_Category']].values,  # Ensure alignment
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +  # Category
                "Award Year: %{x}<br>" +  # Correct syntax
                "Average Publication Timegap: %{y} years<br>" +  # Correct syntax
                "<extra></extra>"  # Hide the trace name
            )
        )

        # Add scatter trace to figure
        fig.add_trace(scatter_trace)

        
        # Perform linear regression for the trendline
        X = subset["Prize0_AwardYear"].values.reshape(-1, 1)
        y = subset["PublicationTimegap"].values
        
        if len(X) > 1:  # Only perform regression if we have enough points
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Add the trendline to the plot
            fig.add_trace(go.Scatter(
                x=subset["Prize0_AwardYear"], 
                y=y_pred,  
                mode='lines', 
                name=f"{category} Trendline",
                line=dict(color=colors[category], dash='dot'),
                customdata=data[['Prize0_Category']],
                hovertemplate=(
                    "<b>%{x}</b><br>" +  # Year
                    "Average Publication Timegap: %{y:.1f} years<br>"  # life expectancy
                    "<extra>%{customdata[0]}</extra>"  # Hide the trace name
                )
            ))

    # Add the new red line from df_lifeexpectancy
    fig.add_trace(go.Scatter(
        x=df_lifeexpectancy["Year"], 
        y=df_lifeexpectancy["World"], 
        mode='lines', 
        name="Life Expectancy World",
        line=dict(color=th.FIG_INK_LIGHT, width=2),  # ink line; swaps to light in dark mode
        hovertemplate=(
            "<b>%{x}</b><br>" +  # Years
            "Life Expectancy: %{y:.1f} years<br>"  # life expectancy
            "<extra>World</extra>"  # Hide the trace name
        )
    ))

    # Add the new red line from df_lifeexpectancy
    fig.add_trace(go.Scatter(
        x=df_lifeexpectancy["Year"], 
        y=df_lifeexpectancy["Europe"], 
        mode='lines', 
        name="Life Expectancy Europe",
        line=dict(color=th.SPECTRUM_LIGHT["Physics"], width=2),  # spectrum blue
        hovertemplate=(
            "<b>%{x}</b><br>" +  # Years
            "Life Expectancy: %{y:.1f} years<br>"  # life expectancy
            "<extra>Europe</extra>"  # Hide the trace name
        )

    ))

    # Update layout
    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        xaxis_title="Year of Nobel Prize Award",
        yaxis_title="Average Time Gap (years)",
        showlegend=True,
        margin={"r":0,"t":0,"l":0,"b":0},
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
        ),
        # title=dict(
        #     text = "3.b: Timegap between Seminal Paper and Nobel Prize with Trendlines",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        # width = 1000,
        # height = 600,
        autosize=True
    )

    # Hover label styling
    fig.update_layout(
        # hovermode="x",
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )

    # Range slider
    fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True, 
        ),  
        type="linear",
        )
    )

    fig.update_xaxes(rangeslider_thickness = 0.05)  # sets the slider height to 0.5% of the plot height


    return fig


def prepare_data_age(data, categories, gender, timerange, timerange_field):
    """
    Prepares data for analyzing the average age of Nobel laureates at the time 
    of receiving their award.

    Args:
        data (DataFrame): The input dataset of laureates.
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        categories (str): categories filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        DataFrame: A grouped DataFrame with average age at award by year 
        and prize categories.
    """

    # standard filter
    df_filtered = standard_filter(data, categories, gender, timerange, timerange_field)

    # Copy the relevant columns to avoid warnings
    df_age = df_filtered[["Prize0_AwardYear", "Prize0_Category", "BirthDate", "Prize0_DateAwarded"]].copy()

    # Ensure that the columns are in datetime format
    df_age["BirthDate"] = pd.to_datetime(df_age["BirthDate"], format="%Y-%m-%d", errors='coerce')
    df_age["Prize0_DateAwarded"] = pd.to_datetime(df_age["Prize0_DateAwarded"], format="%Y-%m-%d", errors='coerce')

    # Subtract the dates to get the timedelta
    df_age["Age_at_Award_Days"] = df_age["Prize0_DateAwarded"] - df_age["BirthDate"]

    # Convert days to years only if the dtype is timedelta64[ns]
    df_age["Age_at_Award_Years"] = np.floor(df_age["Age_at_Award_Days"].dt.days / 365.25)

    # Drop rows with NaN values
    df_age = df_age.dropna(subset=["Age_at_Award_Years"])

    # Remove rows where Age_at_Award_Years is 0 or empty
    df_age = df_age[df_age["Age_at_Award_Years"] != 0]

    # Group by Award Year and Category, and calculate the mean age
    df_age_grouped = df_age.groupby(["Prize0_AwardYear", "Prize0_Category"]).agg(Avg_Age_at_Award_Years=("Age_at_Award_Years", "mean")).reset_index()

    return df_age_grouped


def generate_scatterbox_age(data=df_laureates, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generates a scatter plot with trendlines showing the average age of Nobel 
    laureates at the time of award, grouped by year and prize categories.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        categories (str): categories filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        plotly.graph_objects.Figure: A scatter plot with trendlines, 
        categorized by prize type, showing average age trends.
    """

    data = prepare_data_age(data, categories, gender, timerange, timerange_field)

    colors = {
        "Physics": cf.c_physics, 
        "Chemistry":cf.c_chemistry, 
        "Medicine": cf.c_medicine,
        "Economic Sciences": cf.c_economics,
        "Literature": cf.c_literature,
        "Peace": cf.c_peace
    }

    fig = px.scatter(
        data, 
        x="Prize0_AwardYear", 
        y="Avg_Age_at_Award_Years", 
        color="Prize0_Category",
        # trendline="ols",  # Ordinary Least Squares trendline
        labels={
            "Prize0_AwardYear": "Year of Award",
            "Avg_Age_at_Award_Years": "Average Age at Award",
            "Prize0_Category": "Prize Category"
        },
        color_discrete_map=colors,  # Custom color mapping
    )

    categories = data['Prize0_Category'].unique()
    for category in categories:
        subset = data[data['Prize0_Category'] == category]
        
        # Linear regression fÃƒÂ¼r die Trendline
        from sklearn.linear_model import LinearRegression
        X = subset["Prize0_AwardYear"].values.reshape(-1, 1)
        y = subset["Avg_Age_at_Award_Years"].values
        
        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Trendline hinzufÃƒÂ¼gen
            fig.add_trace(go.Scatter(
                x=subset["Prize0_AwardYear"], 
                y=y_pred,  
                mode='lines', 
                name=f"{category} Trendline",
                line=dict(color=colors[category], dash='dot'),
            ))

    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r":0,"t":50,"l":0,"b":0},
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
        ),        
        # title=dict(
        #     text = "3.c: Laureate Age at Time of Award (with trendlines)",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        autosize= True
        ),

    # Hover label template
    fig.update_traces(
        customdata=data[['Prize0_Category']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +  # Category
            "Award Year: %{x}<br>" + # Year
            "Average Age: %{y:.1f}"  # Age
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


    # Range slider
    fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True, 
        ),  
        type="linear",
        )
    )

    fig.update_xaxes(rangeslider_thickness = 0.05)  # sets the slider height to 0.5% of the plot height

    return fig


def generate_heatmap_age(data=df_laureates, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award"):
    """
    Generates a heatmap showing the average age of Nobel laureates at the 
    time of award, categorized by prize and year.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        categories (str): categories filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        plotly.graph_objects.Figure: A heatmap showing average laureate 
        ages by award year and prize categories.
    """

    data = prepare_data_age(data, categories, gender, timerange, timerange_field)

    data = data.pivot_table(
        index="Prize0_Category", 
        columns="Prize0_AwardYear", 
        values="Avg_Age_at_Award_Years"
    )
    
    fig = px.imshow(
        data, 
        labels={
            "x": "Year of Award",
            "y": "Prize Category",
            "color": "Average Age at Award"
        },
        x=data.columns,
        y=data.index,
        color_continuous_scale= cf.c_colorscale_red,
    )

    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        margin={"r":0,"t":50,"l":0,"b":0},
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
        ),
        # title=dict(
        #     text = f"3.c: Laureate Age at Time of Award",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),
        autosize= True
    )

    fig.update_traces(
        customdata=data.values,  # Pass heatmap values for hover display
        hovertemplate=(
            "<b>%{y}</b><br>" +  # Prize category
            "Award Year: %{x}<br>" +  # Year of award
            "Average Age: %{customdata:.1f}<br>" +  # Show the average age, formatted to 1 decimal place
            "<extra></extra>"  # Hide trace info
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


    # Range slider
    fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True, 
        ),  
        type="linear",
        )
    )

    fig.update_xaxes(rangeslider_thickness = 0.05)  # sets the slider height to 0.5% of the plot height


    return fig
