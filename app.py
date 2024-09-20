##################################################################################################
# Library Imports
##################################################################################################

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly
import pandas as pd
import dash_ag_grid as dag
import os
import json
import numpy as np



##################################################################################################
# General Settings
##################################################################################################

main_brand_color='rgb(84, 67, 12)'
secondary_brand_color='rgb(255, 67, 12)'
brand_color_plot_background='#FAFAFA'

colorscale_1 = [
        [0, 'rgb(201, 188, 147)'],  # Light tone of the brand color at value 0
        [0.2, 'rgb(160, 130, 35)'],  # A mid-tone brown at value ~87
        [0.5, 'rgb(84, 67, 12)'],   # Original base color at value 100
        [0.7, 'rgb(140, 40, 40)'],  # Transition to darker red at value ~202
        [1, 'rgb(100, 0, 0)']       # Darker red at the maximum value 289
        ],

colorscale_2_cont = ['#002CF0', '#9E261F', '#5D3870', '#047507', '#00F0E5', '#F0AA06']
colorscale_3_cont = ['rgb(201, 188, 147)', 'rgb(160, 130, 35)', 'rgb(84, 67, 12)', 'rgb(140, 40, 40)', 'rgb(100, 0, 0)']



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

max_prize_count = df_nobelprizes_percountry['Count'].max()



##################################################################################################
# Plots
##################################################################################################


# Sample Graphs
# ================================================================================================
fig1 = px.line(x=[1, 2, 3], y=[10, 20, 30], title="Line Chart")
fig2 = px.bar(x=["A", "B", "C"], y=[5, 3, 6], title="Bar Chart")
fig3 = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 1, 9], mode='markers', name='Scatter'))



# Rotatable Globe (Nobel Laureates per Country of Birth)
# ================================================================================================

def generate_globe_plot(data):
    
    fig = go.Figure(data=go.Choropleth(
        locations=data['ISO3'],
        z=data['Count'],
        # colorscale='Blues',
        colorscale=colorscale_3_cont,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='No of Laureates',
    )) 
   
    fig.update_layout(      
        template='seaborn',

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = main_brand_color,
        ),

        # showlegend = True,

        title=dict(
            text = "Nobel Laureates by Country of Birth",
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




# Places of Birth of Nobel Laureates
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
            color=main_brand_color,  # Color of the marker (your brand color)
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
            color = main_brand_color,
        ),
        
        title=dict(
            text = f"Places of {city.capitalize()} of Nobel Laureates",
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

fig_1d = generate_scatterbox_plot(df_laureates)


##################################################################################################
# Dashboard
##################################################################################################

# Initialize the app with Bootstrap stylesheet
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app = dash.Dash(__name__) # loads the theme in assets

import dash_mantine_components as dmc
from dash import Dash, _dash_renderer
_dash_renderer._set_react_version("18.2.0")


# Assume df_laureates, fig_1b, fig_1d, fig2, fig3, and max_prize_count are defined

app = dash.Dash(__name__)

# Defining the grid of AGGrid
grid = dag.AgGrid(
    id="nl-aggrid",
    rowData=df_laureates.to_dict("records"),
    columnDefs=[{"field": i} for i in df_laureates.columns],
)

# Correct usage of Paper with padding
tab1_content = dmc.Paper(
    children=[
        html.H5("Nobel Laureates by Country of Birth", className="card-title"),
        html.Div("The rotatable, zoomable globe below shows the number of laureates born in the respective country. You may use the slider to set lower and upper limits. Note that country borders have changed throughout history, so there is some fuzziness involved.", className="mt-5 mb-5"),
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
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    span=10,
                ),
            ],
            justify="left",
            style={"margin-top": "30px"}
        ),
        html.Hr(style={'border': '1px solid #aaa', 'margin-top': '50px', 'margin-bottom': '50px'}),
        html.H5("Nobel Laureates by City of Birth", className="card-title"),
        html.Div("Shows cities of birth.", className="mt-5 mb-5"),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='cities-map', figure=fig_1d), span=12)
            ]
        ),
        dmc.Grid(
            children=[
                dmc.GridCol(
                    html.Div("Please select location:", style={"margin-top": "10px", "font-weight": "bold"}),
                    span="auto"  # Auto width for the label
                ),
                dmc.GridCol(
                    dcc.Dropdown(
                        id='city-dropdown',
                        options=[
                            {'label': 'City of Birth', 'value': 'birth'},
                            {'label': 'City of Death', 'value': 'death'}
                        ],
                        value='birth',  # Default value
                        clearable=False,
                    ),
                    span=6  # Width for the dropdown
                ),
            ],
            justify="left",
            style={"margin-top": "20px"}
        ),
    ],
    shadow="md",
    radius="md",
    p="lg",  # Correct usage of padding attribute
    className="mt-5",
)

tab2_content = dmc.Paper(
    children=[
        html.H5("Tab 2: Bar Chart", className="card-title"),
        dcc.Graph(figure=fig2),
    ],
    shadow="md",
    radius="md",
    p="lg",  # Correct usage of padding attribute
    className="mt-3",
)

tab3_content = dmc.Paper(
    children=[
        html.H5("Tab 3: Scatter Plot", className="card-title"),
        html.Div("This page shows a scatter plot."),
        dcc.Graph(figure=fig3),
    ],
    shadow="md",
    radius="md",
    p="lg",  # Correct usage of padding attribute
    className="mt-3",
)

tab4_content = dmc.Paper(
    children=[
        html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This page shows a scatter plot."),
        html.Div([grid]),
    ],
    shadow="md",
    radius="md",
    p="lg",  # Correct usage of padding attribute
    className="mt-3",
)

# Define app layout
# app.layout = dmc.Container(
app.layout = dmc.MantineProvider(
    children=[
        dmc.GridCol(
            children=[
                dmc.GridCol(html.H1("Nobel Laureate Data Dashboard v10.35", className="text-center mt-5 mb-5"), span=12)
            ]
        ),
        dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab(dmc.Text("Nationality"), value="tab1"),
                        dmc.TabsTab(dmc.Text("Bar Chart"), value="tab2"),
                        dmc.TabsTab(dmc.Text("Scatter Plot"), value="tab3"),
                        dmc.TabsTab(dmc.Text("Full Data"), value="tab4"),
                    ]
                ),
                dmc.TabsPanel(tab1_content, value="tab1"),
                dmc.TabsPanel(tab2_content, value="tab2"),
                dmc.TabsPanel(tab3_content, value="tab3"),
                dmc.TabsPanel(tab4_content, value="tab4"),
            ],
            value="tab1",  # Default selected tab
            id="tabs",
        ),
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

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5085)
