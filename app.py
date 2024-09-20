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

# Country Populations
df_pop = pd.read_excel("df_population.xlsx")

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
            text = "1.b: Nobel Prizes per Country (Country of Birth)",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 1,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
            pad=dict(t = 20, b = 20)
        ),

        width=1200, 
        height=800,

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

        width = 1200,
        height = 800,

        mapbox=dict(
            style="carto-positron",  # Free CartoDB Positron map style
            zoom=1,  # Set default zoom level
            center=dict(lat=20, lon=0)  # Default map center
        ),

    )

    return fig

fig_1c = generate_scatterbox_plot(df_laureates)


# Plot 1.d: Prizes per Country x Population
# ================================================================================================

# Get some relevant columsn from the master table, rename
df_nlpc = df_prizes[['BirthCountryNow', 'LaureateGender', 'Prize0_AwardYear']]
df_nlpc.columns = ['Country', 'Gender', 'Year']

# Merge the counts list and the ISO3 list
df_nlpc = pd.merge(df_nlpc, df_iso, on="Country")

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

# Add a small constant to ensure minimum bubble size
df_nlpc_complete['AdjustedSize'] = df_nlpc_complete['RunningSum'] + 10

# Create a column for the text labels
df_nlpc_complete['Text'] = df_nlpc_complete.apply(lambda row: f"{row['Country']}: {int(row['RunningSum'])}", axis=1)


# clean POP data
df_pop.pop("ISO3")

# Function to convert the population values accurately
def convert_population_accurate(value):
    if isinstance(value, str):
        if 'B' in value:
            return float(value.replace('B', '')) * 1_000_000_000
        elif 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'k' in value:
            return float(value.replace('k', '')) * 1_000
        else:
            try:
                return int(value)
            except ValueError:
                return value  # Return the original value if conversion fails
    return value  # Return the original value if it's not a string

# Apply the conversion function to all columns except the first one ('country')
for column in df_pop.columns[1:]:
    df_pop[column] = df_pop[column].apply(convert_population_accurate)

# Melt the population DataFrame
df_pop_melted = df_pop.melt(id_vars=["country"], var_name="year", value_name="population")
df_pop_melted.columns=["Country", "Year", "Population"]

df_nlpc_complete_population = df_nlpc_complete.merge(df_pop_melted, how='left', on=["Country", "Year"])

df_nlpc_complete_population['Prizes'] = pd.to_numeric(df_nlpc_complete_population['RunningSum'], errors='coerce')
df_nlpc_complete_population['Population'] = pd.to_numeric(df_nlpc_complete_population['Population'], errors='coerce')
df_nlpc_complete_population["PrizesPerPop"] = (df_nlpc_complete_population["RunningSum"] / df_nlpc_complete_population["Population"])
df_nlpc_complete_population["PrizesPer1MPop"] = (df_nlpc_complete_population["RunningSum"] / df_nlpc_complete_population["Population"])*1000000


# Log transform the Population and add the 4th root of PrizesPer1MPop
df_nlpc_complete_population_log = df_nlpc_complete_population
df_nlpc_complete_population_log['LogPopulation'] = np.log(df_nlpc_complete_population_log['Population'] + 1) # Adding 1 to avoid log(0)
df_nlpc_complete_population_log['SQRT4ofPrizesPer1MPop'] = np.power(df_nlpc_complete_population_log['PrizesPer1MPop'], (1/4)) # forth root, gives a nice scaling


# plot settings

df_nlpc_complete_population_log.loc[:, 'PopulationNormalized'] = min_max_normalize(df_nlpc_complete_population['Population'])
y_range_max = df_nlpc_complete_population_log['SQRT4ofPrizesPer1MPop'].max()*1.1

min_size = 0.2
bubblesize = np.maximum(df_nlpc_complete_population_log['PopulationNormalized']*10, min_size)*3


# generate the plot

def generate_population_bubbles(data):

    fig = px.scatter(data, 
                    x="LogPopulation", 
                    y="SQRT4ofPrizesPer1MPop", 
                    size=bubblesize, 
                    color="Country",
                    hover_name="Country", 
                    animation_frame="Year", 
                    range_x=[np.log(50_000), np.log(2_000_000_000)],  # Adjust range for log scale
                    range_y=[0, y_range_max],  # Adjust range for 1M scale                
                    text="Text",
                    hover_data={
                        'Year': True,
                        #'Country': True,
                        'Prizes': True,
                        'Population': True,
                        'SQRT4ofPrizesPer1MPop': ':.2f',  # Format to 2 decimal places
                        },
                    )

    fig.update_layout(

            template='plotly_white',
            plot_bgcolor=brand_color_plot_background,

            margin={"r":0,"t":60,"l":0,"b":0},

            font=dict(
                family = 'Rubik, sans-serif',
                size = 11,
                color = brand_color_main,
            ),
            
            # showlegend = True,

            title=dict(
                text = "5.a: Nobel Prizes x Country of Birth x Population: 1901-2023",
                font=dict(size = 20),
                x = 0,                            # Left align the title
                xanchor = 'left',                 # Align to the left edge
                y = 0.97,                         # Adjust Y to position title above the map
                yanchor = 'top',                  # Anchor at the top of the title box
            ),

            width=1200, 
            height=800,

    )

    # Define tick values and labels for the x-axis (population)
    tickvals_x = np.log([50_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_500_000_000, 2_000_000_000])
    ticktext_x = ['50k', '100k', '1M', '10M', '100M', '1.5B', '2.0B']

    # Define tick values and labels for the y-axis (normalized prizes per population)
    tickvals_y = np.power([0, 1, 2, 3, 4, 5, 10, 20, 50], (1/4))
    ticktext_y = ['0', '1', '2', '3', '4', '5', '10', '20', '50']

    # Update x-axis and y-axis to show original values
    fig.update_xaxes(tickvals=tickvals_x, ticktext=ticktext_x, title="Population (log-scale)")
    fig.update_yaxes(tickvals=tickvals_y, ticktext=ticktext_y, title="Prizes Per Population (4th root)")

    # Adjust the position of the text labels
    fig.update_traces(textposition='top center')

    return fig

fig_5a = generate_population_bubbles(df_nlpc_complete_population_log)


# Plot 5.b: Prizer per Country Stacked Bar Chart
# ================================================================================================

df_pivot = df_nlpc_complete.pivot(index='Year', columns='Country', values='Prizes').fillna(0)
df_pivot = df_pivot.reset_index()


def generate_prizerpercountry(data):

    fig = px.bar(df_pivot, x='Year', y=df_pivot.columns[1:], title='Number of Nobel Prizes by Country per Year',
    labels={'value': 'Number of Prizes', 'variable': 'Country'})

    # Update the layout to stack bars
    fig.update_layout(barmode='stack', xaxis_title='Year', yaxis_title='Number of Prizes')

    fig.update_layout(

            template='plotly_white',
            plot_bgcolor=brand_color_plot_background,

            margin={"r":0,"t":60,"l":0,"b":0},

            font=dict(
                family = 'Rubik, sans-serif',
                size = 11,
                color = brand_color_main,
            ),
            
            # showlegend = True,

            title=dict(
                text = "5.b: Nobel Prizes per Country (Individual Years)",
                font=dict(size = 20),
                x = 0,                            # Left align the title
                xanchor = 'left',                 # Align to the left edge
                y = 0.97,                         # Adjust Y to position title above the map
                yanchor = 'top',                  # Anchor at the top of the title box
            ),

            width=1200, 
            height=600,

    )

    return fig

fig_5b = generate_prizerpercountry(df_pivot)


# Plot 5.c: Prizer per Country Stacked Bar Chart (Running Sum)
# ================================================================================================

df_pivot_rs = df_nlpc_complete.pivot(index='Year', columns='Country', values='RunningSum').fillna(0)
df_pivot_rs = df_pivot_rs.reset_index()


def generate_prizespercountry_rs(data):

    fig = px.bar(df_pivot_rs, x='Year', y=df_pivot.columns[1:], title='Number of Nobel Prizes by Country per Year',
                labels={'value': 'Number of Prizes', 'variable': 'Country'})

    # Update the layout to stack bars
    fig.update_layout(barmode='stack', xaxis_title='Year', yaxis_title='Number of Prizes')

    fig.update_layout(

        template='plotly_white',
        plot_bgcolor=brand_color_plot_background,

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
        ),
        
        # showlegend = True,

        title=dict(
            text = "Nobel Prizes per Country (Running Sum)",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 0.97,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
        ),

        width=1200, 
        height=600,

    )
    
    return fig

fig_5c = generate_prizespercountry_rs(df_pivot_rs)



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

        width=1200, 
        height=800,

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

        width = 1200,
        height = 800,
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

        width = 1200,
        height = 800,
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

        width = 1200,
        height = 1800,
        ),

    return fig

fig_4a = generate_migration_dwp(df_movement_dwp, "DegreeCountry", "WorkCountry", "PrizeCountry")


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
        html.Div("This section contains plots related to the Nationality of Nobel Laureates. Actually, most of the time, it refers to the country of birth, which is unique and easy to identify, unlike the actual nationality. Note that quite some Laureates will actually not have the nationaliyt of the country they were born in."),
        dmc.Space(h="xl"),
        html.Div("Some of the plots offer additional selection option, such as date range sliders, or options which data should be displayed."),
        dmc.Space(h="xl"),

        
        # dmc.Stack(
        #     gap=0,
        #     children=[
        #         dmc.Skeleton(h=50, mb="xl"),
        #         dmc.Skeleton(h=8, radius="xl"),
        #         dmc.Skeleton(h=8, my=6),
        #         dmc.Skeleton(h=8, w="70%", radius="xl"),
        #     ],
        # ),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_1b', figure=fig_1b), span=12)
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
    
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_1c', figure=fig_1c), span=12)
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
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_5a', figure=fig_5a), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_5b', figure=fig_5b), span=12)
            ]
        ),

        dmc.Stack([dmc.Space(h="xl"), dmc.Divider(size="lg"), dmc.Space(h="xl")]),

        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_5c', figure=fig_5c), span=12)
            ]
        ),
    ],

    shadow="md",
    radius="md",
    p="lg",
    className="mt-3",
)


# Contents of Tab 2

tab2_content = dmc.Paper(
    children=[
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This tab contains plots related to gender, ethnicity and relgion. I am aware that that these categories are sometimes disputed or disputable, and I havebeen trying to use the in a sensible way."),
        dmc.Space(h="xl"),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_2a', figure=fig_2a), span=12)
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
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("Plots on this tabe relate to the age of Nobel Laureates, as well as to the time gap between their invention/research/seminal paper and the Nobel Prize."),
        dmc.Space(h="xl"),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_3a', figure=fig_3a), span=12)
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
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("This tab contains plots related to the migration of Nobel laureates."),
        dmc.Space(h="xl"),
        dmc.Grid(
            children=[
                dmc.GridCol(dcc.Graph(id='fig_4a', figure=fig_4a), span=12)
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
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("Placeholder Tab"),
        dmc.Space(h="xl"),
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
        # html.H5("Tab 4: Full Data", className="card-title"),
        html.Div("Here you can access the raw data, download it, and see references."),
        dmc.Space(h="xl"),
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
                                dmc.TabsTab("Nationality & Country", value="tab1"),
                                dmc.TabsTab("Gender, Ethnicity & Religion", value="tab2"),
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
    Output('fig_1b', 'figure'),
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
    Output('fig_1c', 'figure'),
    Input('city-dropdown', 'value'),
)
def update_cities_map(selected_city_type): # the passed value here is passed before from the callback automatically
    return generate_scatterbox_plot(df_laureates, selected_city_type)




##################################################################################################
# Running the app
##################################################################################################

# --------------------
# Run the app (locally)
# if __name__ == "__main__":
#     app.run(debug=True, port=5085)

# Run the app on render.com
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Fallback to port 8050 if PORT isn't set
    app.run_server(host='0.0.0.0', port=port, debug=True)

