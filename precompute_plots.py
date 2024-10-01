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
# import dash
# from dash import dcc, html
# from dash import Dash, _dash_renderer
# from dash.dependencies import Input, Output
# import dash_mantine_components as dmc
from sklearn.linear_model import LinearRegression
# _dash_renderer._set_react_version("18.2.0")
import pickle




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
c_red_superlight = '#fbe0e2'
c_red_verydark = '#3A070B'

c_teal_verylight = '#C2EEF3'
c_teal_superlight = '#e1f7f9'
c_teal_verydark = '#072124'

c_lightblue_verylight = '#e9f1f8'
c_lightblue_superlight = '#f4f8fc'


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

# Ethnicity
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';')

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';')

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
    'the Netherlands': 'Netherlands',
    'East Timor': 'Timor-Leste',
    'USA': 'United States'

}

# Call the function
df_laureates = replace_values_in_columns(df_laureates_import, columns_to_modify, replacement_dict)
df_prizes = replace_values_in_columns(df_prizes_import, columns_to_modify, replacement_dict)

df_laureates.to_csv("df_laureates_cleaned.csv", sep=';', encoding="UTF-8")
df_prizes.to_csv("df_prizes_cleaned.csv", sep=';', encoding="UTF-8")

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

df_nobelprizes_percountry.to_csv("df_nobelprizes_percountry.csv", sep=';', encoding="UTF-8")

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
        colorscale=colorscale_teal_to_read,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar=dict(
            title='No of Laureates',
            len=0.4,  # Adjust the height
            x=1,  # Adjust the x position (default is 1, which is far right)
            xanchor='left',  # Align colorbar with its left side at x=0.9
            y=1,  # Center the colorbar vertically
            yanchor='top'  # Align colorbar around its middle at y=0.5
        ),
    )) 

    fig.update_layout(      
        template='plotly_white',

        margin={"r":0,"t":0,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
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
        # height=800,
        autosize=True,

        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='orthographic'
        )
    )

    return fig

# fig_1b = generate_globe_plot(df_nobelprizes_percountry)



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
            size=9,
            color=brand_color_main, 
            opacity=0.8
        ),
        text=data['AwardeeDisplayName'],  # Laureate name (used for hover)
        hoverinfo='text', 
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
        
        # title=dict(
        #     text = f"1.c: Places of {city.capitalize()} of Nobel Laureates",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),

        # width = 1200,
        # height = 800,
        autosize=True,

        mapbox=dict(
            style="carto-positron",  # Free CartoDB Positron map style
            zoom=1,  # Set default zoom level
            center=dict(lat=20, lon=0)  # Default map center
        ),

    )

    return fig

# fig_1c = generate_scatterbox_plot(df_laureates)


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

            # title=dict(
            #     text = "5.a: Nobel Prizes x Country of Birth x Population: 1901-2023",
            #     font=dict(size = 20),
            #     x = 0,                            # Left align the title
            #     xanchor = 'left',                 # Align to the left edge
            #     y = 0.97,                         # Adjust Y to position title above the map
            #     yanchor = 'top',                  # Anchor at the top of the title box
            # ),

            # width=1200, 
            # height=800,
            autosize=True
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

# fig_5a = generate_population_bubbles(df_nlpc_complete_population_log)


# Plot 5.b: Prizer per Country Stacked Bar Chart
# ================================================================================================

df_pivot = df_nlpc_complete.pivot(index='Year', columns='Country', values='Prizes').fillna(0)
df_pivot = df_pivot.reset_index()


def generate_prizerpercountry(data):

    fig = px.bar(df_pivot, x='Year', y=df_pivot.columns[1:],
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

            # title=dict(
            #     text = "5.b: Nobel Prizes per Country (Individual Years)",
            #     font=dict(size = 20),
            #     x = 0,                            # Left align the title
            #     xanchor = 'left',                 # Align to the left edge
            #     y = 0.97,                         # Adjust Y to position title above the map
            #     yanchor = 'top',                  # Anchor at the top of the title box
            # ),

            # width=1200, 
            # height=600,
            autosize=True
    )

    return fig

# fig_5b = generate_prizerpercountry(df_pivot)


# Plot 5.c: Prizer per Country Stacked Bar Chart (Running Sum)
# ================================================================================================

df_pivot_rs = df_nlpc_complete.pivot(index='Year', columns='Country', values='RunningSum').fillna(0)
df_pivot_rs = df_pivot_rs.reset_index()


def generate_prizespercountry_rs(data):

    fig = px.bar(df_pivot_rs, x='Year', y=df_pivot.columns[1:], # title='Number of Nobel Prizes by Country per Year',
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

        # title=dict(
        #     text = "Nobel Prizes per Country (Running Sum)",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 0.97,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        # ),

        # width=1200, 
        # height=600,
        autosize=True

    )
    
    return fig

# fig_5c = generate_prizespercountry_rs(df_pivot_rs)



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

        # title=dict(
        #     text = "2.a: Nobel Prizes to Women per Discipline and Decade",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 0.97,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        # ),

        # width=1200, 
        # height=800,
        autosize=True,

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

# fig_2a = generate_surface_plot(df_prizes_forwomen_perdecade)



# Plot 2.b: Men & Women
# ================================================================================================

# 1: Remove rows with gender "female" or "org"
df_prizes_formen_perdecade_c = df_prizes_pergender_perdecade_b[(df_prizes_pergender_perdecade_b['LaureateGender'] != 'female') & (df_prizes_pergender_perdecade_b['LaureateGender'] != 'org')]

# 2: As we have only men left, delete the column gender
df_prizes_formen_perdecade_d = df_prizes_formen_perdecade_c.drop(columns=['LaureateGender']) 

# 3: Pivot the table, so that decades apear on the x-axis, and disciplines on the y-axis. Counts are now values, with zeroes filled in for empty values.
df_prizes_formen_perdecade = df_prizes_formen_perdecade_d.pivot_table(index='Prize0_Category', columns='Decade', values='Count', fill_value=0)




def generate_surface_plot_mw(data_m, data_w):

    y = ["Chem", "Eco", "Lit", "Med", "Peace", "Phys"]
    x = ["1900", "1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"]


    # Provide the basic parameters
    fig = go.Figure(data=[go.Surface(z=data_w.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7),
                        go.Surface(z=data_m.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7)  # We just added another surface here, for the men.
                        ])

    # And some layout details
    fig.update_layout(
        # title='Nobel Prizes to Men and Women per Discipline and Decade',
        autosize=True,
        # width=1000, 
        # height=800,
        template='plotly_white',
        scene=dict(                             
            xaxis=dict(
                tickvals=x,
                ticktext=x,
                title='Decades',
                autorange='reversed' 
            ),
            yaxis=dict(
                title='Categories'  
            ),
            zaxis=dict(
                title='Number of Prizes to Men & Women'  
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), 
                center=dict(x=0, y=0, z=0),       
                up=dict(x=0, y=0, z=1)           
            ),
        )
    )

    return fig



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
        
        # title=dict(
        #     text = f"3.a: Histogram of Timegap between Seminal Paper and Nobel Prize",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),

        legend_title_text="Prize Category",

        # width = 1200,
        # height = 800,
        autosize=True
        ),
    
    return fig

# fig_3a = generate_timegap_histogram(df_timegap)



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
        line=dict(color=c_literature, width=2)  
    ))

            # Add the new red line from df_lifeexpectancy
    fig.add_trace(go.Scatter(
        x=df_lifeexpectancy["Year"], 
        y=df_lifeexpectancy["Europe"], 
        mode='lines', 
        name="Life Expectancy Europe",
        line=dict(color=c_peace, width=2) 
    ))

    # Update layout
    fig.update_layout(
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
        
        # title=dict(
        #     text = f"3.b: Timegap between Seminal Paper and Nobel Prize with Trendlines",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),

        # width = 1200,
        # height = 800,
        autosize=True
        ),

    return fig


# fig_3b = generate_timegap_trend(df_timegap, df_lifeexpectancy)


# Plot 3.c: Age Scatterplot
# ================================================================================================


# Copy the relevant columns to avoid warnings
df_age = df_laureates[["Prize0_AwardYear", "Prize0_Category", "BirthDate", "Prize0_DateAwarded"]].copy()

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

# Replace phs/med with med
df_age["Prize0_Category"] = df_age["Prize0_Category"].replace("Physiology or Medicine", "Medicine")

# Group by Award Year and Category, and calculate the mean age
df_age_grouped = df_age.groupby(["Prize0_AwardYear", "Prize0_Category"]).agg(Avg_Age_at_Award_Years=("Age_at_Award_Years", "mean")).reset_index()


def generate_age_scatterbox(data):

    colors = {
        "Physics": c_physics, 
        "Chemistry":c_chemistry, 
        "Medicine": c_medicine,
        "Economic Sciences": c_economics,
        "Literature": c_literature,
        "Peace": c_peace
    }

    fig = px.scatter(
        df_age_grouped, 
        x="Prize0_AwardYear", 
        y="Avg_Age_at_Award_Years", 
        color="Prize0_Category",
        trendline="ols",  # Ordinary Least Squares trendline
        labels={
            "Prize0_AwardYear": "Year of Award",
            "Avg_Age_at_Award_Years": "Average Age at Award (Years)",
            "Prize0_Category": "Prize Category"
        },
        color_discrete_map=colors,  # Custom color mapping
    )

    fig.update_layout(

            margin={"r":0,"t":50,"l":0,"b":0},

            font=dict(
                family = 'Rubik, sans-serif',
                size = 14,
                color = brand_color_main,
            ),
            
            # title=dict(
            #     text = f"3.c: Laureate Age at Time of Award (with trendlines)",
            #     font=dict(size = 20),
            #     x = 0,                            # Left align the title
            #     xanchor = 'left',                 # Align to the left edge
            #     y = 1,                         # Adjust Y to position title above the map
            #     yanchor = 'top',                  # Anchor at the top of the title box
            #     pad=dict(t = 20, b = 20)
            # ),

            autosize= True
            ),

    return fig

# fig_3c = generate_age_scatterbox(df_age_grouped)



# Plot 3.d: Age Heatmap
# ================================================================================================

# Pivot the data for heatmap
df_age_pivot = df_age_grouped.pivot_table(
    index="Prize0_Category", 
    columns="Prize0_AwardYear", 
    values="Avg_Age_at_Award_Years"
)

def generate_age_heatmap(data):
    fig = px.imshow(
        data, 
        labels={
            "x": "Year of Award",
            "y": "Prize Category",
            "color": "Average Age at Award (Years)"
        },
        x=data.columns,
        y=data.index,
        color_continuous_scale= colorscale_teal_to_read,
    )

    fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
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
    return fig


# Series 4: Migration
##################################################################################################

# Plot 4.a: Movement Degree - Work - Prize
# ================================================================================================

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
        
        # title=dict(
        #     text = f"4.a: Movement from Locations of Degree / Achievement / Prize",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 1,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        #     pad=dict(t = 20, b = 20)
        # ),

        # width = 1200,
        # height = 1800,
        autosize=True
        ),

    return fig

# fig_4a = generate_migration_dwp(df_movement_dwp, "DegreeCountry", "WorkCountry", "PrizeCountry")


# Plot 4.b: Movement Birth - Prize -  Death
# ==============================================================================================

df_movement_bpd = df_laureates[["BirthCityNow", "BirthCountryNow","BirthContinent","Prize0_Affiliation0_CityNow", "Prize0_Affiliation0_Country","Prize0_Affiliation0_Continent", "DeathCityNow", "DeathCountryNow", "DeathContinent"]]
df_movement_bpd = df_movement_bpd.fillna('None')

def generate_migration_bpd(data, loc1="BirthCountryNow", loc2="Prize0_Affiliation0_Country", loc3="DeathCountryNow"):

    data['color_value'] = data[loc1].factorize()[0]  # Factorize converts categories to unique integers

    fig = go.Figure(data=[go.Parcats(
            dimensions=[
                {'label': 'Birth', 'values': data[loc1]},
                {'label': 'Work', 'values': data[loc2]},
                {'label': 'Death', 'values': data[loc3]}
            ],
            line={
                'color': data['color_value'],  # Use the mapped numerical values for coloring
                'colorscale': colorscale_palette,
                'shape': 'hspline'  # hspline is the attribute for curved lines
            },
            hoveron='color', # Hover on color
            hoverinfo='all', # Display all available information on hover
            arrangement='freeform', # Allows for dragging categories without snapping to a grid
        )])

    fig.update_layout(title="Nobel Laureate Migration (Countries)", width=1400, height=1800)

    return fig

# fig_4b = generate_migration_bpd(df_movement_bpd)


# Series 6: Fun facts
##################################################################################################

df_prizemoney = df_prizes[["Prize0_AwardYear", "Prize0_Category", "Prize0_Portion", "Prize0_Amount", "Prize0_AmountAdjusted_"]].copy(deep=True)

df_prizemoney.loc[:, "Prize0_Portion"] = df_prizemoney["Prize0_Portion"].apply(lambda x: float(eval(x)))

df_prizemoney.loc[:, "PrizeAmountShared"] = df_prizemoney["Prize0_Amount"] * df_prizemoney["Prize0_Portion"]

df_prizemoney.loc[:, "PrizeAmountAdjustedShared"] = df_prizemoney["Prize0_AmountAdjusted_"] * df_prizemoney["Prize0_Portion"]

df_prizemoney_rs = df_prizemoney.groupby("Prize0_AwardYear").agg({
    "PrizeAmountShared": "sum",
    "PrizeAmountAdjustedShared": "sum"
}).reset_index()


df_prizemoney_rs["CumulativePrizeAmountShared"] = df_prizemoney_rs["PrizeAmountShared"].cumsum()
df_prizemoney_rs["CumulativePrizeAmountAdjustedShared"] = df_prizemoney_rs["PrizeAmountAdjustedShared"].cumsum()

totalprizeamount = max_value = df_prizemoney_rs["CumulativePrizeAmountAdjustedShared"].max()

def generate_prizemoney_linechart(data):

    # Create a line chart with Plotly
    fig = go.Figure()

    # Add the line for the running sum
    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountShared"],
        mode='lines+markers',
        name='Cumulative Prize Amount (SEK)',
        line=dict(color=brand_color_alt),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountAdjustedShared"],
        mode='lines+markers',
        name='Cumulative Prize Amount Inflation Adjusted (SEK)',
        line=dict(color=brand_color_alt2),
        marker=dict(size=8)
    ))
    # Customize layout
    fig.update_layout(

        xaxis_title="Year",
        yaxis_title="Cumulative Prize Amount (SEK)",
        template="plotly_white",
        plot_bgcolor=c_lightblue_superlight,

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
        ),
        
        # showlegend = True,

        title=dict(
            text = "Total Nobel Prize Award Amount in SEK (Running Sum)",
            font=dict(size = 20),
            x = 0,                            # Left align the title
            xanchor = 'left',                 # Align to the left edge
            y = 0.97,                         # Adjust Y to position title above the map
            yanchor = 'top',                  # Anchor at the top of the title box
        ),

        legend=dict(
            x=0.02,            # Position from left; adjust for padding
            y=0.93,            # Position from top; adjust for padding
            xanchor='left',    # Anchor legend by the left
            yanchor='top',     # Anchor legend by the top
            bgcolor='rgba(255, 255, 255, 0.5)', # Optional: Background color with transparency
            bordercolor='rgba(0, 0, 0, 0.2)',   # Optional: Border color
            borderwidth=1,     # Optional: Border width
            # borderpad=10,      # Padding around the legend box
            font=dict(size=10) # Font size of legend text
        ),

        autosize=True
    )

    return fig

# fig_6a = generate_prizemoney_linechart(df_prizemoney_rs)


# GENDER

ds_gender_counts = df_laureates['LaureateGender'].value_counts()

# Convert the Series to a DataFrame
df_gender_counts = ds_gender_counts.to_frame()

# Optionally reset the index if you want to make 'LaureateGender' a separate column
df_gender_counts = df_gender_counts.reset_index()

# Rename the columns for clarity
df_gender_counts.columns = ['Gender', 'Count']

# df_gender_counts.head()

def generate_gender_donut(data):
    labels = data['Gender'].tolist()  # Extract 'LaureateGender' column as labels
    values = data['Count'].tolist()           # Extract 'Count' column as values
    colors = [c_teal, c_magenta]

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(
                    colors=colors  # Use the colors in the order of the labels
                ),
                showlegend=False,
                textinfo='label',  # Show only labels, no percentages or values
                textposition='inside',  # Force text inside the slices
                #insidetextorientation='horizontal'  # Keeps text horizontal inside slices
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text='Gender', 
                x=0.5, 
                y=0.5,
                font_size=20,
                showarrow=False,
                xanchor="center"
            )
        ],
        # width = 260,
        # height = 260,
        margin=dict(l=10, r=10, t=10, b=10),  # Reduce the margins
    )

    return fig


# ETHNICITY

ds_ethnicity_counts = df_ethnicity['Ethnicity'].value_counts()

# Convert the Series to a DataFrame
df_ethnicity_counts = ds_ethnicity_counts.to_frame()

# Optionally reset the index if you want to make 'LaureateGender' a separate column
df_ethnicity_counts = df_ethnicity_counts.reset_index()

# Rename the columns for clarity
df_ethnicity_counts.columns = ['Ethnicity', 'Count']

def generate_ethnicity_donut(data):
    labels = data['Ethnicity'].tolist()  # Extract 'LaureateGender' column as labels
    values = data['Count'].tolist()           # Extract 'Count' column as values
    colors = [c_teal, c_red, c_brown, c_orange, c_lightblue, c_magenta, c_darkmagenta]  # Example color codes: darkmagenta, teal, orange, pink

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(
                    colors=colors  # Use the colors in the order of the labels
                ),
                showlegend=False,
                textinfo='label',  # Show only labels, no percentages or values
                textposition='inside',  # Force text inside the slices
                #insidetextorientation='horizontal'  # Keeps text horizontal inside slices
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text='Ethnicity', 
                x=0.5, 
                y=0.5,
                font_size=20,
                showarrow=False,
                xanchor="center"
            )
        ],
        # width = 260,
        # height = 260,
        margin=dict(l=10, r=10, t=10, b=10),  # Reduce the margins

    )
    
    return fig


# RELIGION

ds_religion_counts = df_religion['Religion'].value_counts()

# Convert the Series to a DataFrame
df_religion_counts = ds_religion_counts.to_frame()

# Optionally reset the index if you want to make 'LaureateGender' a separate column
df_religion_counts = df_religion_counts.reset_index()

# Rename the columns for clarity
df_religion_counts.columns = ['Religion', 'Count']


def generate_religion_donut(data):
    labels = data['Religion'].tolist()  # Extract 'LaureateGender' column as labels
    values = data['Count'].tolist()           # Extract 'Count' column as values
    colors = [c_teal, c_red, c_orange, c_lightblue, c_magenta, c_darkmagenta]  # Example color codes: darkmagenta, teal, orange, pink

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(
                    colors=colors  # Use the colors in the order of the labels
                ),
                showlegend=False,
                textinfo='label',  # Show only labels, no percentages or values
                textposition='inside',  # Force text inside the slices
                #insidetextorientation='horizontal'  # Keeps text horizontal inside slices
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text='Religion', 
                x=0.5, 
                y=0.5,
                font_size=20,
                showarrow=False,
                xanchor="center"
            )
        ],
        # width = 260,
        # height = 260,
        margin=dict(l=10, r=10, t=10, b=10),  # Reduce the margins
    )
    
    return fig


##################################################################################################
# Generate and Save Plots
##################################################################################################

if __name__ == "__main__":
    # Generating the plots
    def generate_plots():
        fig_1b = generate_globe_plot(df_nobelprizes_percountry)
        fig_1c = generate_scatterbox_plot(df_laureates)
        fig_5a = generate_population_bubbles(df_nlpc_complete_population_log)
        fig_5b = generate_prizerpercountry(df_pivot)
        fig_5c = generate_prizespercountry_rs(df_pivot_rs)
        fig_2a = generate_surface_plot(df_prizes_forwomen_perdecade)
        fig_2b = generate_surface_plot_mw(df_prizes_formen_perdecade, df_prizes_forwomen_perdecade)
        fig_2c = generate_gender_donut(df_gender_counts)
        fig_2d = generate_ethnicity_donut(df_ethnicity_counts)
        fig_2e = generate_religion_donut(df_religion_counts)
        fig_3a = generate_timegap_histogram(df_timegap)
        fig_3b = generate_timegap_trend(df_timegap, df_lifeexpectancy)
        fig_3c = generate_age_scatterbox(df_age_grouped)
        fig_3d = generate_age_heatmap(df_age_pivot)
        fig_4a = generate_migration_dwp(df_movement_dwp, "DegreeCountry", "WorkCountry", "PrizeCountry")
        fig_4b = generate_migration_bpd(df_movement_bpd, "BirthCountryNow", "Prize0_Affiliation0_Country", "DeathCountryNow")
        fig_6a = generate_prizemoney_linechart(df_prizemoney_rs)
        
        return {
            'fig_1b': fig_1b, 
            'fig_1c': fig_1c, 
            'fig_5a': fig_5a, 
            'fig_5b': fig_5b,
            'fig_5c': fig_5c, 
            'fig_2a': fig_2a, 
            'fig_2b': fig_2b, 
            'fig_3a': fig_3a, 
            'fig_3b': fig_3b, 
            'fig_3c': fig_3c, 
            'fig_3d': fig_3d, 
            'fig_4a': fig_4a, 
            'fig_4b': fig_4b,
            'fig_6a': fig_6a,
            'fig_2c': fig_2c,
            'fig_2d': fig_2d,
            'fig_2e': fig_2e
            }


    # Generate plots and save to a file
    precomputed_plots = generate_plots()
    with open('precomputed_plots.pkl', 'wb') as f:
        pickle.dump(precomputed_plots, f)


    print("Plots precomputed.")
