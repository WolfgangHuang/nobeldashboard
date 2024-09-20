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
_dash_renderer._set_react_version("18.2.0")



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
c_red_verydark = '#3A070B'

c_teal_verylight = '#C2EEF3'
c_teal_verydark = '#072124'


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
# General Functions
##################################################################################################

# Define a Min-Max normalization function to manipulate the bubble size based on population.
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())



##################################################################################################
# Data Import
##################################################################################################

# Get the path of the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to this location
os.chdir(current_dir)

#Import from local project folder
df_laureates_import = pd.read_csv('df_laureates.csv', sep=';')

# Same as laureates, but the two-time-winners are listed twice
df_prizes_import = pd.read_csv('df_prizes.csv', sep=';')

# Timegap Seminal Paper and Prize
df_timegap = pd.read_csv('df_prize-publication-timegap.csv', sep=';', encoding='UTF-8')


# Average life expectancy data
df_lifeexpectancy = pd.read_excel('df_life-expectancy.xlsx')

# Degree - Work - Prize Movement
df_movement_dwp = pd.read_excel('df_degree_institutions_work.xlsx')
df_movement_dwp = df_movement_dwp.fillna('None')

# ISO3 list
df_iso = pd.read_csv('countries_iso2_iso3.csv', sep=';')


##################################################################################################
# Data Cleaning
##################################################################################################


# There are some non-conventional country names (reference: geonames.org) in the data. These will be replaced now.

def replace_values_in_columns(df, columns, replacement_dict):
    """
    Replace values in the given columns of a DataFrame according to the provided dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column headers where replacements should be made.
    replacement_dict (dict): A dictionary where keys are the values to be replaced, and values are the replacements.

    Returns:
    pd.DataFrame: DataFrame with the replaced values.
    """
    # Ensure columns exist in the DataFrame
    for column in columns:
        if column in df.columns:
            # Use pandas replace method to replace values in the column according to the dictionary
            df[column] = df[column].replace(replacement_dict)
    
    return df


# Columns to modify
columns_to_modify = ['BirthCountryNow', 
                     'DeathCountryNow', 
                     'Prize0_Affiliation0_CountryNow', 
                     'Prize0_Affiliation1_CountryNow', 
                     'Prize0_Residence0_CountryNow', 
                     'Prize0_Residence1_CountryNow', 
                     'Prize1_Affiliation0_CountryNow', 
                     'Prize0_Affiliation2_CountryNow', 
                     'Prize0_Affiliation3_CountryNow']

# Dictionary for replacement
replacement_dict = {
    'Czech Republic': 'Czechia',
    'Faroe Islands (Denmark)': 'Denmark',
    'Northern Ireland': 'United Kingdom',
    'Scotland': 'United Kingdom',
    'Guadeloupe, France': 'Guadeloupe',
    'The Netherlands': 'Netherlands',
    'East Timor': 'Timor-Leste',
    'USA': 'United States'

}

# Call the function
df_laureates = replace_values_in_columns(df_laureates_import, columns_to_modify, replacement_dict)
df_prizes = replace_values_in_columns(df_prizes_import, columns_to_modify, replacement_dict)



##################################################################################################
# Count Laureates per Country
##################################################################################################

# Count unique values in "BirthCountryNow"
ds_nobelprizes_percountry = df_laureates['BirthCountryNow'].value_counts()

# The result is a Pandas series object; but we prefer it to be a Pandas data frame. Let's convert it.
df_nobelprizes_percountry = ds_nobelprizes_percountry.reset_index()

# Rename the columns
df_nobelprizes_percountry.columns = ['Country', 'Count']

# Merge the counts list and the ISO3 list
df_nobelprizes_percountry = pd.merge(df_nobelprizes_percountry, df_iso, on="Country")

df_nobelprizes_percountry.head(100)

max_prize_count = df_nobelprizes_percountry['Count'].max()



##################################################################################################
# Plots
##################################################################################################

# Series 1: Country and City of Birth on Maps
##################################################################################################

# Plot 1a: Simple Map
# ================================================================================================
# not included, because it doesn't appear in the dashboard



# Plot 1b: Rotatable Globe (Nobel Laureates per Country of Birth)
# ================================================================================================

def generate_globe_plot(data):
    
    fig = go.Figure(data=go.Choropleth(
        locations=data['ISO3'],
        z=data['Count'],
        # colorscale='Blues',
        colorscale=colorscale_teal_to_read,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='No of Laureates',
    )) 
   
    fig.update_layout(      
        template='plotly_white',

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
        ),

        # showlegend = True,

        title=dict(
            text = "1.b: Nobel Prizes per Country (Globe)",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        width=800, 
        height=600,

        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='orthographic'
        )
    )

    return fig

fig_1b = generate_globe_plot(df_nobelprizes_percountry)



# Plot 1.c: Places of Birth of Nobel Laureates
# ================================================================================================

def generate_scatterbox_plot(data, city="birth"):

    # Function to add small jitter to coordinates to avoid overlap
    def add_jitter(coordinates, scale=0.05):
        return coordinates + np.random.uniform(-scale, scale, size=len(coordinates))

    if city == "birth":
        # Add jitter to latitude and longitude
        latitudes = add_jitter(data['BirthCityNowLat'])
        longitudes = add_jitter(data['BirthCityNowLon'])
        hover_text = [
            f"Name: {name}<br>City: {city}<br>Country: {country}<br>Date: {date}"
            for name, city, country, date in zip(data['AwardeeDisplayName'], data['BirthCity'], data['BirthCountryNow'], data['BirthDate'])
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
    fig = go.Figure(go.Scattermapbox(
        lat=latitudes,    # Jittered Latitude coordinates
        lon=longitudes,    # Jittered Longitude coordinates
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9,   # Marker size
            color=brand_color_main,  # Color of the marker (your brand color)
            opacity=0.8
        ),
        text=data['AwardeeDisplayName'],  # Laureate name (used for hover)
        hoverinfo='text',  # Tooltip content
        hovertext=hover_text  # Use dynamically generated hover text
    ))

    # Update layout of the map
    fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
        ),
        
        title=dict(
            text = f"1.c: Places of {city.capitalize()} of Nobel Laureates",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        width = 800,
        height = 600,

        mapbox=dict(
            style="carto-positron",  # Free CartoDB Positron map style
            zoom=1,  # Set default zoom level
            center=dict(lat=20, lon=0)  # Default map center
        ),

    )

    return fig

fig_1c = generate_scatterbox_plot(df_laureates)



# Series 2: Gender and Ethnicity
##################################################################################################



# Plot 2.a: Women
# ================================================================================================

# Get required data from main df
df_prizes_pergender_perdecade_a = df_prizes[['LaureateGender', 'Prize0_AwardYear', 'Prize0_Category']].copy()

# Calculate the decade and drop the year
df_prizes_pergender_perdecade_a['Decade'] = (df_prizes_pergender_perdecade_a['Prize0_AwardYear'] // 10) * 10  # The //-operator rounds down.
df_prizes_pergender_perdecade_a.drop(columns=['Prize0_AwardYear'], inplace=True) 

# Count the genders per decade per discipline (using the groupby-function)
df_prizes_pergender_perdecade_b = df_prizes_pergender_perdecade_a.groupby(['Decade', 'LaureateGender', 'Prize0_Category']).size().reset_index(name='Count')

# 1: Remove rows with gender "male" or "org"
df_prizes_forwomen_perdecade_c = df_prizes_pergender_perdecade_b[(df_prizes_pergender_perdecade_b['LaureateGender'] != 'male') & (df_prizes_pergender_perdecade_b['LaureateGender'] != 'org')]

# 2: As we have only women left, delete the column gender
df_prizes_forwomen_perdecade_d = df_prizes_forwomen_perdecade_c.drop(columns=['LaureateGender']) 

# 3: Pivot the table, so that decades apear on the x-axis, and disciplines on the y-axis. Counts are now values, with zeroes filled in for empty values.
df_prizes_forwomen_perdecade = df_prizes_forwomen_perdecade_d.pivot_table(index='Prize0_Category', columns='Decade', values='Count', fill_value=0)


def generate_surface_plot(data):

    y = ["Chem", "Eco", "Lit", "Med", "Peace", "Phys"]
    x = ["1900", "1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"]

    fig = go.Figure(data=[go.Surface(
                            z=data.values, 
                            y=y, 
                            x=x, 
                            colorscale=brand_colorscale_main, 
                            showscale=False, 
                            opacity=0.7)
                        ]
                    )

    fig.update_layout(      

        template='plotly_white',
                
        margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
        ),
        
        # showlegend = True,

        title=dict(
            text = "2.a: Nobel Prizes to Women per Discipline and Decade",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 0.97,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
        ),

        width=800, 
        height=600,

        scene=dict(
            xaxis=dict(
                tickvals=x,
                ticktext=x,
                title='Decades',
                autorange='reversed'  # Reverse the x-axis  
            ),
            yaxis=dict(
                title='Disciplines'  
            ),
            zaxis=dict(
                title='Number of Prizes to Women'  
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Adjust these values to change camera position
                center=dict(x=0, y=0, z=0),        # Adjust these values to change focus point
                up=dict(x=0, y=0, z=1)             # Usually set to the Z-axis for upward direction
            ),
        )
    )

    return fig

fig_2a = generate_surface_plot(df_prizes_forwomen_perdecade)




# Series 3: Age & Time
##################################################################################################


# Plot 3.a: Timegap
# ================================================================================================

def generate_timegap_histogram(data, category="all"):
    fig = px.histogram(data, x="Timegap", color="Prize0_Category", 
                             opacity=1,
                             color_discrete_map={
                                 "Physics": c_physics, 
                                 "Chemistry":c_chemistry, 
                                 "Medicine": c_medicine
                             },
                            nbins=100,  # Set number of bins to 1 per gap,
                            title="Histogram of Timegap Between Seminal Paper and Nobel Prize",
                            labels={
                                "Timegap": "Time Gap (years)",   # X-axis label
                                "count": "Number of Prizes"   # Y-axis label
                            },
                             barmode="group" )
    
    fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
        ),
        
        title=dict(
            text = f"3.a: Histogram of Timegap between Seminal Paper and Nobel Prize",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        legend_title_text="Prize Category",

        width = 800,
        height = 600,
        ),
    
    return fig

fig_3a = generate_timegap_histogram(df_timegap)



# Plot 3.b: Timegap with Trendlines
# ================================================================================================

def generate_timegap_trend(data, df_lifeexpectancy, category="all"):
    fig = go.Figure()

    categories = data['Prize0_Category'].unique()

    colors = {
        "Physics": c_physics, 
        "Chemistry":c_chemistry, 
        "Medicine": c_medicine
    }

    # Adding scatter points for each category
    for category in categories:
        subset = data[data['Prize0_Category'] == category]
        
        # Scatter points for each category
        fig.add_trace(go.Scatter(
            x=subset["Prize0_AwardYear"], 
            y=subset["Timegap"],  
            mode='markers',  
            name=category,
            marker=dict(color=colors[category], size=8)
        ))
        
        # Perform linear regression for the trendline
        X = subset["Prize0_AwardYear"].values.reshape(-1, 1)
        y = subset["Timegap"].values
        
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
                line=dict(color=colors[category], dash='dot')
            ))

            # Add the new red line from df_lifeexpectancy
    fig.add_trace(go.Scatter(
        x=df_lifeexpectancy["Year"], 
        y=df_lifeexpectancy["World"], 
        mode='lines', 
        name="Life Expectancy World",
        line=dict(color=c_literature, width=2)  # Red line for the new data
    ))

            # Add the new red line from df_lifeexpectancy
    fig.add_trace(go.Scatter(
        x=df_lifeexpectancy["Year"], 
        y=df_lifeexpectancy["Europe"], 
        mode='lines', 
        name="Life Expectancy Europe",
        line=dict(color=c_peace, width=2)  # Red line for the new data
    ))

    # Update layout
    fig.update_layout(
        title="Average Timegap per Year by Category with Linear Regression Trendlines and Life Expectancy",
        xaxis_title="Year of Nobel Prize Award",
        yaxis_title="Average Time Gap (years)",
        showlegend=True
    )

    fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
        ),
        
        title=dict(
            text = f"3.b: Timegap between Seminal Paper and Nobel Prize with Trendlines",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        width = 1000,
        height = 600,
        ),

    return fig


fig_3b = generate_timegap_trend(df_timegap, df_lifeexpectancy)




# Series 4: Migration
##################################################################################################

def generate_migration_dwp(data, loc1="DegreeCountry", loc2="WorkCountry", loc3="PrizeCountry"):

    data['color_value'] = data['PrizeCountry'].factorize()[0]  # Factorize converts categories to unique integers


    data = df_movement_dwp
    data['color_value'] = data['PrizeCountry'].factorize()[0]  # Factorize converts categories to unique integers

    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            {'label': 'Degree', 'values': data[loc1]},
            {'label': 'Work', 'values': data[loc2]},
            {'label': 'Prize', 'values': data[loc3]}
        ],
        line={
            'color': data['color_value'],  # Use the mapped numerical values for coloring
            'colorscale': colorscale_palette,
            'shape': 'hspline'  # hspline is the attribute for curved lines
        },
        hoveron='color', # Hover on color
        hoverinfo='all', # Display all available information on hover
        arrangement='freeform' # Allows for dragging categories without snapping to a grid
        
    )])

    fig.update_layout(

        # margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
        ),
        
        title=dict(
            text = f"4.a: Movement from Locations of Degree / Achievement / Prize",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        width = 1600,
        height = 1400,
        ),

    return fig

fig_4a = generate_migration_dwp(df_movement_dwp, "DegreeCity", "WorkCountry", "PrizeCityCountry")


##################################################################################################
# Dashboard
##################################################################################################

import dash_mantine_components as dmc
from dash import Dash, _dash_renderer
_dash_renderer._set_react_version("18.2.0")


# app = Dash(external_stylesheets=dmc.styles.ALL)

# app = Dash(external_stylesheets=["assets/dmc_styles.css"] + dmc.styles.ALL)

app = Dash(
    external_stylesheets=[
        "assets/dmc_styles.css",  # Your custom CSS
        dmc.styles.ALL           # Mantine styles
    ]
)

# ---------------------

# Defining the grid of AGGrid: The full data table
full_table = dag.AgGrid(
    id="nl-aggrid",
    rowData=df_laureates.to_dict("records"),
    columnDefs=[{"field": i} for i in df_laureates.columns],
)

# Contents of Tab 1

tab1_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='globe-plot', figure=fig_1b), span=12)
            ]
        ),
        dmc.Grid(
            children=[
                dmc.GridCol(
                    dcc.RangeSlider(
                        id='prize-slider',
                        min=0,
                        max=max_prize_count,  # Dynamic maximum value based on data
                        step=1,
                        value=[0, max_prize_count],  # Default range from 0 to max
                        marks={i: str(i) for i in range(0, int(max_prize_count) + 1, 50)},  # Custom marks
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="dmc-bar dmc-thumb",
                    ),
                    span=5
                ),
            ],
            justify="left",
            style={"margin-top": "30px"}
        ),
        
        #html.Hr(style={'border': '1px solid #aaa', 'margin-top': '50px', 'margin-bottom': '50px'}),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),
    
    
        
        html.H5("Nobel Laureates by City of Birth", className="card-title"),
        html.Div("Shows cities of birth.", className="mt-5 mb-5"),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='cities-map', figure=fig_1c), span=12)
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
                    style={"width": "200px"}  # Optional: set width for dropdown
                ),
            ],
            gap="md",  # Adjusts the space between the label and the dropdown
            align="flex-start",  # Align items to the left
            style={"margin-top": "20px"}
        )
    ],

    shadow="md",
    radius="md",
    p="lg",
    className="mt-3",
)


# Contents of Tab 2

tab2_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='3dsurface', figure=fig_2a), span=12)
            ]
        ),


    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)

# Contents of Tab 3

tab3_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='timegap', figure=fig_3a), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        html.Div("This page shows a scatter plot."),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='timegaptrends', figure=fig_3b), span=12)
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
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='timegap', figure=fig_4a), span=12)
            ]
        ),

        # dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        # html.Div("This page shows a scatter plot."),
        # dmc.Grid(
        #     children=[
        #         dmc.GridCol(dcc.Graph(id='timegaptrends', figure=fig_4b), span=12)
        #     ]
        # ),

    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)

# Contents of Tab 5

tab5_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        # dmc.Grid(
        #     children=[
        #         dmc.GridCol(dcc.Graph(id='timegap', figure=fig_3a), span=12)
        #     ]
        # ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),


    ],
    shadow="md",
    radius="md",
    p="lg", 
    className="mt-3",
)



tabdata_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        html.Div([full_table]),
    ],
    shadow="md",
    radius="md",
    p="lg",  # Correct usage of padding attribute
    className="mt-3",
)


# Defining the main layout

app.layout = dmc.MantineProvider(
    children=[
        dmc.Container(
            children=[
                dmc.Grid(
                    children=[
                        dmc.GridCol(html.H1("Nobel Laureate Data Dashboard v12.25", className="text-left mt-5 mb-5"), span=12)
                    ]
                ),
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("Nationality", value="tab1"),
                                dmc.TabsTab("Gender, Ethnicity, Religion", value="tab2"),
                                dmc.TabsTab("Time & Age", value="tab3"),
                                dmc.TabsTab("Migration", value="tab4"),
                                dmc.TabsTab("X", value="tab5"),
                                dmc.TabsTab("Full Data", value="tabdata"),
                            ]
                        ),
                        dmc.TabsPanel(tab1_content, value="tab1"),
                        dmc.TabsPanel(tab2_content, value="tab2"),
                        dmc.TabsPanel(tab3_content, value="tab3"),
                        dmc.TabsPanel(tab4_content, value="tab4"),
                        dmc.TabsPanel(tab5_content, value="tab5"),
                        dmc.TabsPanel(tabdata_content, value="tabdata"),
                    ],
                    value="tab1",  # Default selected tab
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
    Output('globe-plot', 'figure'),
    [Input('prize-slider', 'value')]
)
def update_globe(selected_range):
    # Filter the data based on the selected range
    filtered_data = df_nobelprizes_percountry[
        (df_nobelprizes_percountry['Count'] >= selected_range[0]) &
        (df_nobelprizes_percountry['Count'] <= selected_range[1])
    ]
    # Generate and return the updated globe plot
    return generate_globe_plot(filtered_data)

# Cities of Birth
@app.callback(
    Output('cities-map', 'figure'),
    Input('city-dropdown', 'value'),
)
def update_cities_map(selected_city_type): # the passed value here is passed before from the callback automatically
    return generate_scatterbox_plot(df_laureates, selected_city_type)




##################################################################################################
# Runnign the app
##################################################################################################

# --------------------
# Run the app (locally)
# if __name__ == "__main__":
#     app.run(debug=True, port=5085)

# Run the app on render.com
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Fallback to port 8050 if PORT isn't set
    app.run_server(host='0.0.0.0', port=port, debug=True)

