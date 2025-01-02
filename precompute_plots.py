##################################################################################################
# Library Imports
##################################################################################################

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
# import matplotlib.pyplot as plt
import yfinance as yf
import time

##################################################################################################
# Color Settings
##################################################################################################

brand_color_plot_background='#FEFEFA'

c_brown = '#47382a'
c_brown_verylight = '#F0EBE6'
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

c_pie1 = '#67d6e0'
c_pie2 = '#c9ddee'
c_pie3 = '#ec6570'
c_pie4 = '#ffbc82'
c_pie5 = '#fff3a6'
c_pie6 = '#da81d1'
c_pie7 = '#e7a2bf'
c_pie8 = '#ffcce4'
c_pie0 = '#b59b82'

colorscale_palette = [c_teal, c_lightblue, c_red, c_orange, c_yellow, c_darkmagenta, c_magenta, c_pink]
colorscale_palette_light = [c_pie1, c_pie2, c_pie3, c_pie4, c_pie5, c_pie6, c_pie7, c_pie8]

colorscale_red = [c_red_verylight, c_red_verydark]
colorscale_teal = [c_teal_verylight, c_teal_verydark]
colorscale_teal_log = ["#E1F7F9", "#67D6E0", "#67D6E0", "#186F77", "#13595F", "#114D53"]
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
df_laureates_import = pd.read_csv('df_laureates.csv', sep=';', index_col=0)
df_laureates_corrections = pd.read_csv('df_laureates_corrections.csv', sep=';', index_col=0)

# Timegap
df_timegap = pd.read_csv('df_prize-publication-timegap.csv', sep=';', encoding='UTF-8', index_col=0)

# Average life expectancy data
df_lifeexpectancy = pd.read_excel('df_life-expectancy.xlsx')

# Country Populations
df_pop = pd.read_excel("df_population.xlsx")

# Degree - Work - Prize Movement
df_movement_dwp = pd.read_csv('df_degree_institutions_work.csv', sep=';', encoding='UTF-8', index_col=0)
df_movement_dwp = df_movement_dwp.fillna('None')

# Ethnicity
df_ethnicity = pd.read_csv('df_ethnicity.csv', sep=';', index_col=0)

# Religion
df_religion = pd.read_csv('df_religion.csv', sep=';', index_col=0)

# ISO3 list
df_iso = pd.read_csv('countries_iso2_iso3.csv', sep=';')

# Fields
df_fields = pd.read_csv("df_fields.csv", sep=";")

lastyearincluded=2024
# TODO: Determine year dynamically

# print(df_laureates_import.columns)

##################################################################################################
# Data Cleaning
##################################################################################################

# update the csv/df from the Nobel API with corrected information that has been compiled manually

df_laureates_import.update(df_laureates_corrections)


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



##################################################################################################
# Data Enrichment
##################################################################################################

# Add data on religion
df_laureates = pd.merge(df_laureates, df_religion[["Religion", "ReligionSubgroup"]], left_index=True, right_index=True, how="left")

# Add data on ethnicity
df_laureates = pd.merge(df_laureates, df_ethnicity[["Ethnicity"]], left_index=True, right_index=True, how="left")

# Add data on publication-prize-timegap
df_laureates = pd.merge(df_laureates, df_timegap[["PublicationYear", "PublicationSource", "PublicationTimegap"]], left_index=True, right_index=True, how="left")

# Add data on degree-work-prize movement
df_laureates = pd.merge(df_laureates, df_movement_dwp[["ParCatDegreeInstitution", "ParCatDegreeCity", "ParCatDegreeCountry", "ParCatDegreeCityCountry", "ParCatWorkInstitution", "ParCatWorkCity", "ParCatWorkCountry", "ParCatWorkCityCountry", "ParCatPrizeInstitution", "ParCatPrizeCity", "ParCatPrizeCountry", "ParCatPrizeCityCountry"]], left_index=True, right_index=True, how="left")


##################################################################################################
# Generate df_prizes
##################################################################################################

# Step 1: Filter rows where laureates have multiple prizes
# df_multiple_prizes = df_laureates.dropna(subset=['Prize1_AwardYear', 'Prize2_AwardYear'], how='all')
df_prize1 = df_laureates.dropna(subset=['Prize1_AwardYear']).copy(deep=True)
df_prize2 = df_laureates.dropna(subset=['Prize2_AwardYear']).copy(deep=True)

# Step 2: Identify the Prize1... and Prize2.. columns
prize0_columns = [col for col in df_laureates.columns if col.startswith('Prize0_')]
prize1_columns = [col for col in df_laureates.columns if col.startswith('Prize1_')]
prize2_columns = [col for col in df_laureates.columns if col.startswith('Prize2_')]

# # For second prize (Prize1_)
df_prize1.drop(columns=prize0_columns, inplace=True)  # Drop existing Prize0 columns before renaming Prize1_ to Prize0_
df_prize1.drop(columns=prize2_columns, inplace=True)  # Drop existing Prize2 columns before renaming Prize1_ to Prize0_
renamed_columns = [col.replace('Prize1_', 'Prize0_') for col in df_prize1.columns if col.startswith('Prize1_')]
df_prize1.rename(columns={old: new for old, new in zip(df_prize1.columns[df_prize1.columns.str.startswith('Prize1_')], renamed_columns)}, inplace=True)

# # For third prize (Prize2_)
df_prize2.drop(columns=prize0_columns, inplace=True)  # Drop existing Prize0 columns before renaming Prize2_ to Prize0_
df_prize2.drop(columns=prize1_columns, inplace=True)  # Drop existing Prize1 columns before renaming Prize2_ to Prize0_
renamed_columns = [col.replace('Prize2_', 'Prize0_') for col in df_prize2.columns if col.startswith('Prize2_')]
df_prize2.rename(columns={old: new for old, new in zip(df_prize2.columns[df_prize2.columns.str.startswith('Prize2_')], renamed_columns)}, inplace=True)

# # Step 4: Concatenate the original laureates DataFrame with the new prize rows
df_prizes = pd.concat([df_laureates, df_prize1, df_prize2], ignore_index=True)
df_prizes.drop(columns=prize1_columns, inplace=True)  # Drop Prize1 columns, which are now obsolete
df_prizes.drop(columns=prize2_columns, inplace=True)  # Drop Prize2 columns, which are now obsolete
df_prizes['Prize0_AwardYear'] = df_prizes['Prize0_AwardYear'].astype('Int64') # Convert from float to nullable int


##################################################################################################
# Create Reduced version of lists
##################################################################################################

# Saving the corrected lists with all columns
df_laureates_enriched_full_clean = df_laureates.copy()
df_prizes_enriched_full_clean = df_laureates.copy()

# Drop all unnecessary columns - these are used nowhere
df_laureates.drop(
    inplace=True,
    columns=[
        "LaureateNameKnown",
        "LaureateNameFull",
        "LaureateNamePenOriginal",
        "Filename",
        "OrganisationNameNative",
        "OrganisationAcronym",
        "WikidataID",
        "WikidataURL",
        "Prize0_SortOrder",
        "Prize0_Affiliation0_NameNative",
        "Prize0_Affiliation1_Name",
        "Prize0_Affiliation1_NameNow",
        "Prize0_Affiliation1_NameNative",
        "Prize0_Affiliation1_City",
        "Prize0_Affiliation1_CityNow",
        "Prize0_Affiliation1_CityLatitude",
        "Prize0_Affiliation1_CityLongitude",
        "Prize0_Affiliation1_Country",
        "Prize0_Affiliation1_CountryNow",
        "Prize0_Affiliation1_CountryLat",
        "Prize0_Affiliation1_CountryLon",
        "Prize0_Affiliation1_Continent",
        "Prize0_Residence0_City",
        "Prize0_Residence0_CityNow",
        "Prize0_Residence0_Country",
        "Prize0_Residence0_CountryNow",
        "Prize0_Residence0_Continent",
        "Prize0_Residence1_City",
        "Prize0_Residence1_CityNow",
        "Prize0_Residence1_Country",
        "Prize0_Residence1_CountryNow",
        "Prize0_Residence1_Continent",
        "Prize0_Affiliation2_Name",
        "Prize0_Affiliation2_NameNow",
        "Prize0_Affiliation2_NameNative",
        "Prize0_Affiliation2_City",
        "Prize0_Affiliation2_CityNow",
        "Prize0_Affiliation2_CityLatitude",
        "Prize0_Affiliation2_CityLongitude",
        "Prize0_Affiliation2_Country",
        "Prize0_Affiliation2_CountryNow",
        "Prize0_Affiliation2_CountryLat",
        "Prize0_Affiliation2_CountryLon",
        "Prize0_Affiliation2_Continent",
        "Prize0_Affiliation3_Name",
        "Prize0_Affiliation3_NameNow",
        "Prize0_Affiliation3_NameNative",
        "Prize0_Affiliation3_City",
        "Prize0_Affiliation3_CityNow",
        "Prize0_Affiliation3_CityLatitude",
        "Prize0_Affiliation3_CityLongitude",
        "Prize0_Affiliation3_Country",
        "Prize0_Affiliation3_CountryNow",
        "Prize0_Affiliation3_CountryLat",
        "Prize0_Affiliation3_CountryLon",
        "Prize0_Affiliation3_Continent"
    ]
)

# Drop all unnecessary columns - these are used nowhere
df_prizes.drop(
    inplace=True,
    columns=[
        "Prize0_Affiliation1_Name",
        "Prize0_Affiliation1_NameNow",
        "Prize0_Affiliation1_NameNative",
        "Prize0_Affiliation1_City",
        "Prize0_Affiliation1_CityNow",
        "Prize0_Affiliation1_CityLatitude",
        "Prize0_Affiliation1_CityLongitude",
        "Prize0_Affiliation1_Country",
        "Prize0_Affiliation1_CountryNow",
        "Prize0_Affiliation1_CountryLat",
        "Prize0_Affiliation1_CountryLon",
        "Prize0_Affiliation1_Continent",
        "Prize0_Residence0_City",
        "Prize0_Residence0_CityNow",
        "Prize0_Residence0_Country",
        "Prize0_Residence0_CountryNow",
        "Prize0_Residence0_Continent",
        "Prize0_Residence1_City",
        "Prize0_Residence1_CityNow",
        "Prize0_Residence1_Country",
        "Prize0_Residence1_CountryNow",
        "Prize0_Residence1_Continent",
        "Prize0_Affiliation2_Name",
        "Prize0_Affiliation2_NameNow",
        "Prize0_Affiliation2_NameNative",
        "Prize0_Affiliation2_City",
        "Prize0_Affiliation2_CityNow",
        "Prize0_Affiliation2_CityLatitude",
        "Prize0_Affiliation2_CityLongitude",
        "Prize0_Affiliation2_Country",
        "Prize0_Affiliation2_CountryNow",
        "Prize0_Affiliation2_CountryLat",
        "Prize0_Affiliation2_CountryLon",
        "Prize0_Affiliation2_Continent",
        "Prize0_Affiliation3_Name",
        "Prize0_Affiliation3_NameNow",
        "Prize0_Affiliation3_NameNative",
        "Prize0_Affiliation3_City",
        "Prize0_Affiliation3_CityNow",
        "Prize0_Affiliation3_CityLatitude",
        "Prize0_Affiliation3_CityLongitude",
        "Prize0_Affiliation3_Country",
        "Prize0_Affiliation3_CountryNow",
        "Prize0_Affiliation3_CountryLat",
        "Prize0_Affiliation3_CountryLon",
        "Prize0_Affiliation3_Continent",
    ]
)

df_laureates_enriched_redux_clean = df_laureates.copy()
df_prizes_enriched_redux_clean = df_laureates.copy()

##################################################################################################
# Count Laureates per Country
##################################################################################################

def count_per_country(data=df_laureates, country="BirthCountryNow"):
    """
    Count occurrences of Countries

    Parameters:
    data: The input DataFrame. Default: df_laureates
    country: Valid column name whose entries shall be counted. Default: BirthCountryNow

    Returns:
    pd.DataFrame: DataFrame with columns "Country" and "Count".
    """

    # Count unique values in "BirthCountryNow"
    ds_count_per_country = data[country].value_counts()

    # The result is a Pandas series object; but we prefer it to be a Pandas data frame. Let's convert it.
    df_count_per_country = ds_count_per_country.reset_index()

    # Rename the columns
    df_count_per_country.columns = ['Country', 'Count']

    # Merge the counts list and the ISO3 list
    df_count_per_country = pd.merge(df_count_per_country, df_iso, on="Country")

    return df_count_per_country


#df_nobelprizes_percountry.to_csv("df_nobelprizes_percountry.csv", sep=';', encoding="UTF-8")
df=count_per_country()
max_prize_count = df['Count'].max()


##################################################################################################
# Standard Filter
##################################################################################################

def define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace):

        # Map chip states to categories
        chip_states = {
            "Physiology or Medicine": chip_medicine,
            "Physics": chip_physics,
            "Chemistry": chip_chemistry,
            "Economic Sciences": chip_economics,
            "Literature": chip_literature,
            "Peace": chip_peace
        }

        # Get categories to include
        # get the "cat" from "cat, checked", which is taken from chip_states, but only for those that are checked/TRUE
        selected_categories = [cat for cat, checked in chip_states.items() if checked]

        return selected_categories



def standard_filter(data, categories="all", gender="all", timerange=None, timerange_field="award"):

    # Replace short handles with lists
    if categories == "all":
        categories=["Physiology or Medicine", "Physics", "Chemistry", "Economic Sciences", "Literature", "Peace"]
    elif categories == "sci":
        categories=["Physiology or Medicine", "Physics", "Chemistry", "Economic Sciences"]
    elif categories == "natsci":
        categories=["Physiology or Medicine", "Physics", "Chemistry"]
    else:
        categories = categories

    # Appyl category filter
    df_filtered = data[data["Prize0_Category"].isin(categories)]


    # Apply gender filter if not "all"
    if gender.lower() != "all":
        df_filtered = df_filtered[df_filtered["LaureateGender"] == gender]
    else:
        df_filtered = df_filtered
    
    # Apply year range Filter if not empty
    if timerange is None:
        df_filtered = df_filtered

    else:
        if timerange_field == "birth":
            datefield = pd.to_datetime(data['BirthDate'], errors='coerce').dt.year.astype(int)
        elif timerange_field == "death":
            datefield = pd.to_datetime(data['DeathDate'], errors='coerce').dt.year.astype(int)
        else:
            datefield = data['Prize0_AwardYear'].astype(int)
            
        df_filtered = df_filtered[
            (df_filtered[datefield] >= timerange[0]) &
            (df_filtered[datefield] <= timerange[1])
        ]

    return df_filtered


def replace_country_designations(country):
    country = country.replace("birth", "BirthCountryNow")
    country = country.replace("affiliation", "Prize0_Affiliation0_CountryNow")
    country = country.replace("death", "DeathCountryNow")
    return country


##################################################################################################
# Plots
##################################################################################################

# Geography
##################################################################################################

# Plot: Choropleth Globe Countries
# ================================================================================================

def generate_choroplethglobe(data=df_laureates, country="birth", category="all", gender="all", year_range=None, year_range_field="award"):
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
    - year_range (list): The start and end year of range
        Default: None
    - year_range_field (str): The year field/column the range is being applied to.
        - "birth"
        - "award"
        - "death"
        Default: "award"

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The Plotly figure object representing the globe.
    """

    # Filter
    df_filtered = standard_filter(data, category, gender, year_range, year_range_field)

    # Count
    country = replace_country_designations(country)
    data = count_per_country(df_filtered, country)

    # Create figure
    fig = go.Figure(data=go.Choropleth(
        locations=data['ISO3'],
        z=data['Count'],
        colorscale=colorscale_teal_log,
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


# Plot: Scatter Mapbox Cities
# ================================================================================================

def generate_map_cities(data=df_laureates, city="birth", gender="all", category="all"):
    """
    Generates a Scatter Mapbox with the number of Nobel Laureates per City.

    Parameters:
    - data (pd.DataFrame): The input DataFrame. Default: df_laureates.
    - City (str): The column name representing the country. Default: "birth".
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
    data = standard_filter(data, category, gender)

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
        margin={"r":0,"t":0,"l":0,"b":0},
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
        # width = 800,
        # height = 600,
        mapbox=dict(
            style="carto-positron",  # Free CartoDB Positron map style
            zoom=1,  # Set default zoom level
            center=dict(lat=20, lon=0)  # Default map center
        ),
    )

    return fig

# Plot: Prizes per Country x Population  Animated Scatter Bubble Plot
# ================================================================================================

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
    Merges Nobel Laureate prize data with population data to generate a dataset
    suitable for creating population-based bubble charts.

    Args:
        data (pd.DataFrame):
            Should be df_nlpc_complete returned by prepare_data_prizespercountry()
        df_pop (pd.DataFrame, optional):
            A DataFrame containing population data.
            Default is df_pop, using data from gapminder.org
            Note this data has unusual notation, using K, M, B for thousands, millions, billions
            Will not work with standard numbers
        interval(str):
            Specifies the interval between years. Default is 5, so data is shown for every fifth year. If you set it to lower values, the computation will take up to a minute and figure size will go up to 10MB.
    
    Returns:
        tuple:
            - **df_nlpc_complete_population_log (pd.DataFrame)**: 
              The merged and processed DataFrame filtered to every 5th year, containing log-transformed
              population (`LogPopulation`), the fourth root of the prizes per 1 million population 
              (`SQRT4ofPrizesPer1MPop`), and a normalized population column (`PopulationNormalized`).
            - **y_range_max (float)**: 
              The upper limit for the y-axis used in plotting (1.1 times the maximum of `SQRT4ofPrizesPer1MPop`).
            - **bubblesize (pd.Series or np.ndarray)**: 
              Bubble sizes for plotting, scaled by the min-max normalized population, ensuring a minimum size.
    
    Notes:
        - The function calculates a `RunningSum` of prizes per country across years.
        - Population data is melted and merged by ['Country', 'Year'].
        - Converts population values from strings with 'B', 'M', 'k' suffixes into numeric values.
        - Log-transforms population to avoid issues with large population differences.
        - Returns data filtered to every nth year for saving file size.
    """
     
    # Add a small constant to ensure minimum bubble size
    df_nlpc_complete['AdjustedSize'] = df_nlpc_complete['RunningSum'] + 10

    # Create a column for the text labels
    df_nlpc_complete['Text'] = df_nlpc_complete.apply(lambda row: f"{row['Country']}: {int(row['RunningSum'])}", axis=1)

    # clean POP data
    if "ISO3" in df_pop.columns:
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


    # Data Redux, using "interval" - returns every nth year
    df_nlpc_complete_population_log = df_nlpc_complete_population_log[df_nlpc_complete_population_log["Year"] % interval == 0]
    df_nlpc_complete_population_log.loc[:,"LogPopulation"] = df_nlpc_complete_population_log["LogPopulation"].round(4)
    df_nlpc_complete_population_log.loc[:,"SQRT4ofPrizesPer1MPop"] = df_nlpc_complete_population_log["SQRT4ofPrizesPer1MPop"].round(2)


    # plot settings
    df_nlpc_complete_population_log = df_nlpc_complete_population_log.copy()
    df_nlpc_complete_population_log.loc[:,'PopulationNormalized'] = min_max_normalize(df_nlpc_complete_population['Population'])
    y_range_max = df_nlpc_complete_population_log['SQRT4ofPrizesPer1MPop'].max()*1.1

    min_size = 0.2
    df_nlpc_complete_population_log['PopulationNormalized'] = df_nlpc_complete_population_log['PopulationNormalized'].fillna(0)
    bubblesize = np.maximum(df_nlpc_complete_population_log['PopulationNormalized']*10, min_size)*3

    return df_nlpc_complete_population_log, y_range_max, bubblesize




# generate the plot

def generate_bubbles_population(data=df_prizes, country="birth", gender="all", category="all", df_pop=df_pop, interval=5):
    """
    Generates a population-based bubble chart of Nobel Prize data, using Plotly for interactive visualization.
    
    This function:
      1. Applies a standard filter on the input data using `standard_filter()` to limit rows by category and gender.
      2. Prepares the data for bubble plotting by calling `prepare_data_bubbles_population()`; this step merges 
         cumulative prize counts per country/year with population data, and computes numeric transformations 
         (log, min-max normalization).
      3. Creates a Plotly scatter chart (`px.scatter`) with animated frames over the years, scaled by bubble sizes 
         proportional to the (normalized) population. 
    
    Args:
        data (pd.DataFrame, optional):
            Nobel Prize data in a pandas DataFrame. Defaults to `df_prizes` (a global variable in this script). 
            Must contain columns appropriate for filtering by category/gender and merging with population data.
        country (str, optional):
            Country type for merging Nobel data: 
              - "birth" (BirthCountryNow), 
              - "affiliation" (Prize0_Affiliation0_CountryNow), 
              - "death" (DeathCountryNow).
            Defaults to "birth".
        gender (str, optional):
            Filter the data for a specific gender ("male", "female", etc.). Using "all" applies no gender filter.
            Defaults to "all".
        category (str, optional):
            Filter the data for a specific Nobel category ("physics", "chemistry", etc.). Using "all" applies 
            no category filter. Defaults to "all".
        df_pop (pd.DataFrame, optional):
            A DataFrame containing population data by country and year. Defaults to `df_pop` (global variable).
        interval(str):
            Specifies the interval between years. Default is 5, so data is shown for every fifth year. If you set it to lower values, the computation will take up to a minute and figure size will go up to 10MB.

    Returns:
        plotly.graph_objs._figure.Figure:
            A Plotly figure object representing the bubble chart, with an animation slider to step through years.
    
    Notes:
        - Internally calls `standard_filter()` to reduce the dataset based on category/gender.
        - Uses `prepare_data_bubbles_population()` to merge population data, compute transformations (log scale, 
          min-max normalization, etc.), and obtain the bubble sizes.
        - The x-axis is log-transformed population; the y-axis is the 4th root of the "prizes per 1M population."
        - Creates an animation frame for each year (filtered to every 5th year in `prepare_data_bubbles_population()`).
        - Returns a fully configured Plotly figure ready for interactive display or further styling.
    
    """

    data = standard_filter(data, category, gender)
    df_nlpc_complete = prepare_data_prizespercountry(data, country)
    data, y_range_max, bubblesize = prepare_data_bubbles_population(df_nlpc_complete, df_pop, interval)

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
                    #text=None,
                    hover_data={
                        'Year': True,
                        'Prizes': True,
                        'Population': True,
                        'LogPopulation': False,  # Hide automatic hover for x
                        'SQRT4ofPrizesPer1MPop': False,  # Hide automatic hover for y
                        #'size': False,  # Hide the size column
                        'Text': False  # Ensure "Text" is hidden if it's not needed
                    },
    )

    fig.update_layout(
            template='plotly_white',
            plot_bgcolor=brand_color_plot_background,
            margin={"r":0,"t":30,"l":0,"b":0},
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


# Plot: Prizes per Country Stacked Bar Chart
# ================================================================================================

def generate_bar_prizerpercountry(data=df_prizes, country="birth", gender="all", category="all", runningsum=False):
    """
    Generates a stacked bar chart of Nobel Prizes per country per year.
    
    This function:
      1. Applies a standard filter on the input data using `standard_filter()` to limit rows by category and gender.
      2. Prepares the data by calling `prepare_data_stacked_bar()`
      3. Creates a Plotly bar chart. 
    
    Args:
        data (pd.DataFrame, optional):
            Nobel Prize data in a pandas DataFrame. Defaults to `df_prizes` (a global variable in this script). 
            Must contain columns appropriate for filtering by category/gender and merging with population data.
        country (str, optional):
            Country type for merging Nobel data: 
              - "birth" (BirthCountryNow), 
              - "affiliation" (Prize0_Affiliation0_CountryNow), 
              - "death" (DeathCountryNow).
            Defaults to "birth".
        gender (str, optional):
            Filter the data for a specific gender ("male", "female", etc.). Using "all" applies no gender filter.
            Defaults to "all".
        category (str, optional):
            Filter the data for a specific Nobel category ("physics", "chemistry", etc.). Using "all" applies 
            no category filter. Defaults to "all".
        runningsum (boolean):
            True: returns a plot with a running sum
            False: returns a plot with individual values per year (default)

    Returns:
        plotly.graph_objs._figure.Figure:
            A Plotly figure object representing the bar chart.
    
    Notes:
        - Internally calls `standard_filter()` to reduce the dataset based on category/gender.
        - Returns a fully configured Plotly figure ready for interactive display or further styling.
    
    """

    # Filter
    data = standard_filter(data, category, gender)

    # Prepare
    df_nlpc_complete = prepare_data_prizespercountry(data, country)

    # Return running sum (or not)
    if runningsum:
        data = df_nlpc_complete.pivot(index='Year', columns='Country', values='RunningSum').fillna(0)        
    else:
        data = df_nlpc_complete.pivot(index='Year', columns='Country', values='Prizes').fillna(0)
    
    data = data.reset_index()

    fig = px.bar(data, x='Year', y=data.columns[1:],
    labels={'value': 'Number of Prizes', 'variable': 'Country'})

    # Update the layout to stack bars
    fig.update_layout(barmode='stack', xaxis_title='Year', yaxis_title='Number of Prizes')

    fig.update_layout(
            template='plotly_white',
            plot_bgcolor=brand_color_plot_background,
            margin={"r":0,"t":0,"l":0,"b":0},
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




# Prizes per Country Stacked Bar Chart (Running Sum)
# ================================================================================================

# fig=generate_bar_prizerpercountry(runningsum=True)



# Demography
##################################################################################################



# Plot: Prizes to Women and Men per Decade, 3D surface
# ================================================================================================

def prepare_data_3dsurface(data):
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


def generate_surface_plot_mw(data=df_prizes, gender="female"):
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
    # prepare data
    data_w, data_m = prepare_data_3dsurface(data)

    # set axes
    y = ["Chem", "Eco", "Lit", "Peace", "Phys", "Med"]
    x = ["1900", "1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"]

    # filter gender
    if gender == "all":
        data=[go.Surface(z=data_w.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7),
              go.Surface(z=data_m.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7)]
    elif gender =="female":
        data=[go.Surface(z=data_w.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7)]
    elif gender == "male":
        data=[go.Surface(z=data_m.values, y=y, x=x, colorscale=brand_colorscale_main, showscale=False, opacity=0.7)]

    # initiatie figure
    fig = go.Figure(data=data)

    # add some layout details
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
                title='Prizes'  
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), 
                center=dict(x=0, y=0, z=-0.3),       
                up=dict(x=0, y=0, z=2)           
            ),
        )
    )
    return fig


# Plot: Donuts Gender/Ethnicity/Religion
# ================================================================================================

def prepare_data_donuts(data, category, gender, characteristic):
    """
    Prepares data for a donut chart by filtering and aggregating counts 
    based on the specified characteristic.

    Args:
        data (DataFrame): The input dataset containing laureate information.
        category (str): The category filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): The gender filter (e.g., 'male', 'female', or 'all').
        characteristic (str): The characteristic to aggregate by 
            ('gender', 'ethnicity', or 'religion').

    Returns:
        DataFrame: A DataFrame with columns:
            - 'Label': The unique values of the specified characteristic.
            - 'Count': The count of occurrences for each label.
    """

    # Standard filter
    data = standard_filter(data, category, gender)

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


def generate_donut(data=df_laureates, category="all", gender="all", characteristic="gender"):
    """
    Generates a donut chart visualizing the distribution of a specified 
    characteristic (e.g., gender, ethnicity, or religion) among Nobel laureates.

    Args:
        data (DataFrame): The input dataset containing laureate information (default: df_laureates).
        category (str): The category filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): The gender filter (e.g., 'male', 'female', or 'all').
        characteristic (str): The characteristic to visualize 
            ('gender', 'ethnicity', 'religion').

    Returns:
        plotly.graph_objects.Figure: A Plotly donut chart displaying the distribution 
        of the specified characteristic.
    """

    # Prepare data
    data = prepare_data_donuts(data, category, gender, characteristic)

    # Extract labels and values
    labels = data['Label'].tolist()
    values = data['Count'].tolist()
    characteristic_name = characteristic.capitalize()

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(
                    colors=colorscale_palette_light  # Use the colors in the order of the labels
                ),
                showlegend=False,
                textinfo='label',
                textposition='inside',
                insidetextorientation='horizontal'
            )
        ]
    )

    # Update layout
    fig.update_layout(
        annotations=[
            dict(
                text=characteristic_name,
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
                xanchor="center"
            )
        ],
        autosize=True,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig



# Series: Time
##################################################################################################


# Plot: Timegap
# ================================================================================================

def prepare_data_timegap_histogram(data, category, gender, datasource):
    """
    Filters and prepares data for a timegap histogram.

    Args:
        data (DataFrame): The input data containing a 'PublicationSource' column. Usually df_laureates passed from the calling function.
        category (str): The category filter to apply.
        gender (str): The gender filter to apply.
        datasource (str): The data source to filter by 
            ('paper', 'chatgpt', or other).

    Returns:
        DataFrame: The filtered DataFrame with rows where 'PublicationSource' 
        is not empty and matches the specified filters.
    """

    # delete rows where there is no info on the timegap
    data = data[data['PublicationSource'].notna() & (data['PublicationSource'].str.strip() != "")]

    # replace category
    data.loc[:,'Prize0_Category'] = data['Prize0_Category'].replace('Physiology or Medicine', 'Medicine')

    df_filtered = standard_filter(data, category, gender)
    
    if datasource == "paper":
        df_filtered = df_filtered[(df_filtered['PublicationSource']) == "Harvard Dataverse"]

    elif datasource == "chatgpt":
        df_filtered = df_filtered[(df_filtered['PublicationSource']) == "ChatGPT 4c (September 2024)"]
    else:
        df_filtered = df_filtered
    
    return df_filtered


def generate_histogram_timegap(data=df_prizes, category="all", gender="all", datasource="all"):
    """
    Generates a histogram visualizing the time gap between seminal papers 
    and Nobel Prizes.

    Args:
        data (DataFrame): The input dataset (default: df_prizes).
        category (str): Category filter (e.g., 'Physics', 'Chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        datasource (str): Data source filter (e.g., 'paper', 'chatgpt', or 'all').

    Returns:
        plotly.graph_objects.Figure: A histogram of time gaps, colored by 
        prize category, with custom styling and layout.
    """    
    
    data = prepare_data_timegap_histogram(data, category, gender, datasource)

    fig = px.histogram(
        data,
        x="PublicationTimegap",
        color="Prize0_Category", 
        opacity=1,
        color_discrete_map={
            "Physics": c_physics, 
            "Chemistry":c_chemistry, 
            "Medicine": c_medicine
        },
        nbins=100,  # Set number of bins to 1 per gap,
        #title="Histogram of Timegap Between Seminal Paper and Nobel Prize",
        labels={
            "Timegap": "Time Gap (years)",   # X-axis label
            "count": "Number of Prizes"   # Y-axis label
        },
            barmode="group"
    )
    
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
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
    
    return fig



# Plot: Timegap with Trendlines
# ================================================================================================

def generate_scatterbox_timegaptrend(data=df_laureates, df_lifeexpectancy=df_lifeexpectancy, category="all", gender="all"):
    """
    Generates a trend visualization of the time gap between seminal papers 
    and Nobel Prizes, including trendlines and life expectancy comparisons.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        df_lifeexpectancy (DataFrame): Life expectancy data (default: df_lifeexpectancy).
        category (str): Category filter (e.g., 'Physics', 'Chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').

    Returns:
        plotly.graph_objects.Figure: A line and scatter plot with trendlines, 
        categorized by prize, and overlaid with life expectancy trends.
    """

    data = prepare_data_timegap_histogram(data, category, gender, datasource="notapplicable")
    
    fig = go.Figure()

    categories = data['Prize0_Category'].unique()

    colors = {
        "Physics": c_physics, 
        "Chemistry":c_chemistry, 
        "Medicine": c_medicine,
        "Peace": c_peace
    }

    # Adding scatter points for each category
    for category in categories:
        subset = data[data['Prize0_Category'] == category]
        
        # Scatter points for each category
        fig.add_trace(go.Scatter(
            x=subset["Prize0_AwardYear"], 
            y=subset["PublicationTimegap"],  
            mode='markers',  
            name=category,
            marker=dict(color=colors[category], size=8)
        ))
        
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
        xaxis_title="Year of Nobel Prize Award",
        yaxis_title="Average Time Gap (years)",
        showlegend=True,
        margin={"r":0,"t":0,"l":0,"b":0},
        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
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

    return fig


# Plot: Age Scatterplot
# ================================================================================================


def prepare_data_age(data, gender, category):
    """
    Prepares data for analyzing the average age of Nobel laureates at the time 
    of receiving their award.

    Args:
        data (DataFrame): The input dataset of laureates.
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        DataFrame: A grouped DataFrame with average age at award by year 
        and prize category.
    """

    # standard filter
    df_filtered = standard_filter(data, category, gender)

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

    # Replace phs/med with med
    df_age["Prize0_Category"] = df_age["Prize0_Category"].replace("Physiology or Medicine", "Medicine")

    # Group by Award Year and Category, and calculate the mean age
    df_age_grouped = df_age.groupby(["Prize0_AwardYear", "Prize0_Category"]).agg(Avg_Age_at_Award_Years=("Age_at_Award_Years", "mean")).reset_index()

    return df_age_grouped


def generate_scatterbox_age(data=df_laureates, gender="all", category="all"):
    """
    Generates a scatter plot with trendlines showing the average age of Nobel 
    laureates at the time of award, grouped by year and prize category.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        plotly.graph_objects.Figure: A scatter plot with trendlines, 
        categorized by prize type, showing average age trends.
    """

    data = prepare_data_age(data, gender, category)

    colors = {
        "Physics": c_physics, 
        "Chemistry":c_chemistry, 
        "Medicine": c_medicine,
        "Economic Sciences": c_economics,
        "Literature": c_literature,
        "Peace": c_peace
    }

    fig = px.scatter(
        data, 
        x="Prize0_AwardYear", 
        y="Avg_Age_at_Award_Years", 
        color="Prize0_Category",
        trendline="ols",  # Ordinary Least Squares trendline
        labels={
            "Prize0_AwardYear": "Year of Award",
            "Avg_Age_at_Award_Years": "Average Age at Award",
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

    return fig



# Plot: Age Heatmap
# ================================================================================================

def generate_heatmap_age(data=df_laureates, gender="all", category="all"):
    """
    Generates a heatmap showing the average age of Nobel laureates at the 
    time of award, categorized by prize and year.

    Args:
        data (DataFrame): The input dataset of laureates (default: df_laureates).
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').
        category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').

    Returns:
        plotly.graph_objects.Figure: A heatmap showing average laureate 
        ages by award year and prize category.
    """

    data = prepare_data_age(data, gender, category)

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


# Migration
##################################################################################################

# Plot: Movement Degree - Work - Prize
# ================================================================================================

def generate_parcat_migration(data=df_laureates, loc1="ParCatDegreeCountry", loc2="ParCatWorkCountry", loc3="ParCatPrizeCountry", category="all", gender="all", width=1600, height=1400):
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
        
        category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').
        gender (str): Gender filter (e.g., 'male', 'female', or 'all').

    Returns:
        plotly.graph_objects.Figure: A parallel categories diagram showing 
        movement paths, with customizable dimensions and color-coded categories.
    """

    data = standard_filter(data, category, gender)

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
            'colorscale': colorscale_palette,
            'shape': 'hspline'  # hspline is the attribute for curved lines
        },
        hoveron='color', # Hover on color
        hoverinfo='all', # Display all available information on hover
        arrangement='freeform' # Allows for dragging categories without snapping to a grid
        
    )])

    fig.update_layout(
        margin={"r":35,"t":0,"l":50,"b":0},
        font=dict(
            family = 'Rubik, sans-serif',
            size = 14,
            color = brand_color_main,
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
        ),

    return fig


# def generate_parcat_migration_dwp(data=df_laureates, loc1="ParCatDegreeCountry", loc2="ParCatWorkCountry", loc3="ParCatPrizeCountry", category="all", gender="all"):
#     """
#     Generates a parallel categories diagram visualizing Nobel laureates' movement 
#     from degree locations to work locations and prize locations.

#     Args:
#         data (DataFrame): The input dataset of laureates (default: df_laureates).
#         loc1 (str): Column representing the degree location (default: "ParCatDegreeCountry").
#         loc2 (str): Column representing the work location (default: "ParCatWorkCountry").
#         loc3 (str): Column representing the prize location (default: "ParCatPrizeCountry").
#         category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').
#         gender (str): Gender filter (e.g., 'male', 'female', or 'all').

#     Returns:
#         plotly.graph_objects.Figure: A parallel categories diagram showing 
#         movement paths, with customizable dimensions and color-coded categories.
#     """

#     data = standard_filter(data, category, gender)

#     data = data[data['ParCatDegreeCountry'].notna() & (data['ParCatDegreeCountry'].str.strip() != "")]

#     data.loc[:,'color_value'] = data['ParCatPrizeCountry'].factorize()[0]  # Factorize converts categories to unique integers

#     fig = go.Figure(data=[go.Parcats(
#         dimensions=[
#             {'label': 'Degree', 'values': data[loc1]},
#             {'label': 'Work', 'values': data[loc2]},
#             {'label': 'Prize', 'values': data[loc3]}
#         ],
#         line={
#             'color': data['color_value'],  # Use the mapped numerical values for coloring
#             'colorscale': colorscale_palette,
#             'shape': 'hspline'  # hspline is the attribute for curved lines
#         },
#         hoveron='color', # Hover on color
#         hoverinfo='all', # Display all available information on hover
#         arrangement='freeform' # Allows for dragging categories without snapping to a grid
        
#     )])

#     fig.update_layout(
#         margin={"r":35,"t":0,"l":50,"b":0},
#         font=dict(
#             family = 'Rubik, sans-serif',
#             size = 14,
#             color = brand_color_main,
#         ),
#         # title=dict(
#         #     text = "4.a: Movement from Locations of Degree / Achievement / Prize",
#         #     font=dict(size = 20),
#         #     x = 0,                            # Left align the title
#         #     xanchor = 'left',                 # Align to the left edge
#         #     y = 1,                         # Adjust Y to position title above the map
#         #     yanchor = 'top',                  # Anchor at the top of the title box
#         #     pad=dict(t = 20, b = 20)
#         # ),
#         width = 1600,
#         height = 1400,
#         ),

#     return fig

# # Plot: Movement Birth - Prize -  Death
# # ==============================================================================================

# def generate_parcat_migration_bpd(data=df_laureates, loc1="BirthCountryNow", loc2="Prize0_Affiliation0_Country", loc3="DeathCountryNow", gender="all", category="all"):
#     """
#     Generates a parallel categories diagram visualizing Nobel laureates' migration 
#     paths across birth, work, and death locations.

#     Args:
#         data (DataFrame): The input dataset of laureates (default: df_laureates).
#         loc1 (str): Column representing the birth location (default: "BirthCountryNow").
#         loc2 (str): Column representing the work location (default: "Prize0_Affiliation0_Country").
#         loc3 (str): Column representing the death location (default: "DeathCountryNow").
#         gender (str): Gender filter (e.g., 'male', 'female', or 'all').
#         category (str): Category filter (e.g., 'physics', 'chemistry', or 'all').

#     Returns:
#         plotly.graph_objects.Figure: A parallel categories diagram showing 
#         migration paths, with customizable dimensions and color-coded categories.
#     """

#     df_filtered = standard_filter(data, category, gender)

#     data = df_filtered[["BirthCityNow", "BirthCountryNow","BirthContinent","Prize0_Affiliation0_CityNow", "Prize0_Affiliation0_Country","Prize0_Affiliation0_Continent", "DeathCityNow", "DeathCountryNow", "DeathContinent"]]
#     data = data.fillna('None')

#     data['color_value'] = data[loc1].factorize()[0]  # Factorize converts categories to unique integers

#     fig = go.Figure(data=[go.Parcats(
#             dimensions=[
#                 {'label': 'Birth', 'values': data[loc1]},
#                 {'label': 'Work', 'values': data[loc2]},
#                 {'label': 'Death', 'values': data[loc3]}
#             ],
#             line={
#                 'color': data['color_value'],  # Use the mapped numerical values for coloring
#                 'colorscale': colorscale_palette,
#                 'shape': 'hspline'  # hspline is the attribute for curved lines
#             },
#             hoveron='color', # Hover on color
#             hoverinfo='all', # Display all available information on hover
#             arrangement='freeform', # Allows for dragging categories without snapping to a grid
#         )])

#     fig.update_layout(
#         # title="Nobel Laureate Migration (Countries)", 
#         width=1400, 
#         height=1800,
#         # autosize=True
#     )

#     return fig


# Misc
##################################################################################################

def get_conversion_rates():
    """
    Fetches the latest conversion rates for SEK to EUR and SEK to USD using yfinance.

    Returns:
        tuple: the two exchange rates (SEK-EUR, SEK-USD)
    """
    # Fetch SEK to EUR conversion rate
    sek_to_eur_ticker = yf.Ticker("SEKEUR=X")
    sek_to_eur_rate = sek_to_eur_ticker.history(period="1d").iloc[-1]["Close"]

    # Fetch SEK to USD conversion rate
    sek_to_usd_ticker = yf.Ticker("SEKUSD=X")
    sek_to_usd_rate = sek_to_usd_ticker.history(period="1d").iloc[-1]["Close"]

    return sek_to_eur_rate, sek_to_usd_rate

def generate_line_prizemoney(data=df_prizes, currency="EUR"):
    """
    Generates a line chart showing the cumulative Nobel Prize money awarded 
    over time, optionally converted to a specified currency.

    Args:
        data (DataFrame): The input dataset of Nobel Prizes (default: df_prizes).
        currency (str): The currency for prize amounts ('EUR', 'USD', or 'SEK', 
            default: 'EUR').

    Returns:
        plotly.graph_objects.Figure: A line chart with cumulative prize money 
        and inflation-adjusted prize money over the years.
    """

    sek_to_eur_rate, sek_to_usd_rate = get_conversion_rates()

    if currency=="EUR":
        conversionrate = sek_to_eur_rate
        currencyname = "EUR"
    elif currency=="USD":
        conversionrate = sek_to_usd_rate
        currencyname = "USD"
    elif currency=="SEK":
        conversionrate = 1
        currencyname = "SEK"
    

    # Create a deep copy of the relevant columns to avoid the SettingWithCopyWarning
    df_prizemoney = data[["Prize0_AwardYear", "Prize0_Category", "Prize0_Portion", "Prize0_Amount", "Prize0_AmountAdjusted_"]].copy(deep=True)

    # Safely convert Prize0_Portion to numeric using .loc
    df_prizemoney.loc[:, "Prize0_Portion"] = df_prizemoney["Prize0_Portion"].apply(lambda x: float(eval(x)))

    # Safely calculate PrizeAmountShared using .loc
    df_prizemoney.loc[:, "PrizeAmountShared"] = df_prizemoney["Prize0_Amount"] * df_prizemoney["Prize0_Portion"] * conversionrate

    # Safely calculate PrizeAmountAdjustedShared using .loc
    df_prizemoney.loc[:, "PrizeAmountAdjustedShared"] = df_prizemoney["Prize0_AmountAdjusted_"] * df_prizemoney["Prize0_Portion"]  * conversionrate

    df_prizemoney_rs = df_prizemoney.groupby("Prize0_AwardYear").agg({
        "PrizeAmountShared": "sum",
        "PrizeAmountAdjustedShared": "sum"
    }).reset_index()

    # Step 2: Calculate the running sum for each column
    df_prizemoney_rs["CumulativePrizeAmountShared"] = df_prizemoney_rs["PrizeAmountShared"].cumsum()
    df_prizemoney_rs["CumulativePrizeAmountAdjustedShared"] = df_prizemoney_rs["PrizeAmountAdjustedShared"].cumsum()

    data = df_prizemoney_rs
    totalprizeamount = data["CumulativePrizeAmountAdjustedShared"].iloc[-1]

    # Create a line chart with Plotly
    fig = go.Figure()

    # Add the line for the running sum
    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountShared"],
        mode='lines+markers',
        name=f'Cumulative Prize Amount ({currencyname})',
        line=dict(color=brand_color_alt),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountAdjustedShared"],
        mode='lines+markers',
        name=f'Cumulative Prize Amount ({currencyname}) Inflation Adjusted',
        line=dict(color=brand_color_alt2),
        marker=dict(size=8)
    ))
    # Customize layout
    fig.update_layout(

        xaxis_title="Year",
        yaxis_title=f"Cumulative Prize Amount ({currencyname})",
        template="plotly_white",
        plot_bgcolor=brand_color_plot_background,

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'Rubik, sans-serif',
            size = 11,
            color = brand_color_main,
        ),
        
        # showlegend = True,

        # title=dict(
        #     text = "Total Nobel Prize Award Amount (Running Sum)",
        #     font=dict(size = 20),
        #     x = 0,                            # Left align the title
        #     xanchor = 'left',                 # Align to the left edge
        #     y = 0.97,                         # Adjust Y to position title above the map
        #     yanchor = 'top',                  # Anchor at the top of the title box
        # ),

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

    return fig, totalprizeamount




# 2024 Facts
##################################################################################################

# Plot Sunburst 2024
#=================================================================================================

def prepare_data_sunburst(data, year, path):
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

    if year=="last":
        data = data[data['Prize0_AwardYear'] == lastyearincluded]
    else:
        data = data
 
    # Group by Category and Gender to get the counts
    data_count = data.groupby(path).size().reset_index(name='count')
    data_count.replace({'Physiology or Medicine': 'Medicine', 'Economic Sciences': 'Economics', 'United States': 'USA'}, inplace=True)

    return data_count


def generate_sunburst(data=df_laureates, year="all", path=['Prize0_Category', 'LaureateGender', 'BirthCountryNow']):
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

    data = prepare_data_sunburst(data, year, path)

    fig = px.sunburst(
        data, 
        path=path, 
        values='count',
        color='Prize0_Category',
        color_discrete_map={
            "Medicine": c_medicine,
            "Physics": c_physics,
            "Chemistry": c_chemistry,
            "Literature": c_literature,
            "Peace": c_peace,
            "Economics": c_economics,
        }
    )
    return fig



# Plot Movement Splines on Map/Globe
#=================================================================================================

def prepare_data_splines(data, year, category, gender):
   """
   Filters and formats Nobel laureates' data for visualizing movement splines.

   Parameters:
   ----------
   data : pd.DataFrame
      Input DataFrame with laureates' details.
   year : str or int
      Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work.
   category : str
      Nobel Prize category (e.g., 'physics', 'all').
   gender : str
      Gender filter ('male', 'female', 'all').

   Returns:
   -------
   pd.DataFrame
      Processed DataFrame with key columns and valid location data.
   """

   # standard filter
   data = standard_filter(data, category, gender)

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



def generate_globe_movemment(data=df_laureates, year="last", category="all", gender="all"):
    """
    Creates a globe visualization showing Nobel laureates' migration paths.

    Parameters:
    ----------
    data : pd.DataFrame, optional
        Input DataFrame containing laureates' data (default is `df_laureates`).
    year : str or int, optional
        Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work. (default is "last").
    category : str, optional
        Nobel Prize category to filter (e.g., 'physics', default is "all").
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

    colors = colorscale_palette

    data = prepare_data_splines(data, year, category, gender)

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
            color=c_brown,
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
            text=f"{row['AwardeeDisplayName']}<br>Birth: {row['BirthCityNow']}<br>Affiliation: {row['Prize0_Affiliation0_CityNow']}",
            name=f"{row['AwardeeDisplayName']}"
        ))

    # Update layout for global view
    fig.update_layout(
        # title_text="Migration Paths of 2023 Nobel Prize Laureates",
        showlegend=False,
        geo=go.layout.Geo(
            projection_type="orthographic",
            showland=True,
            landcolor=c_brown_verylight,  # Terrain-like land color
            countrycolor=c_brown,  # Darker color for country borders
            countrywidth=0.8,  # Border width
            coastlinecolor=c_brown,  # Darker coastlines
            coastlinewidth=0.5,  # Coastline width
            showlakes=True,
            lakecolor=c_lightblue,  # Same as water color for lakes
            showcountries=True,
            showocean=True,
            oceancolor=c_lightblue_superlight,
            showframe=False,  # Removes the box frame
        ),
        height=900,
        #autosize=True
    )

    return fig


def generate_map_movement(data=df_laureates, year="last", category="all", gender="all"):
    """
    Creates an interactive map visualizing Nobel laureates' migration paths with curved lines.

    Parameters:
    ----------
    data : pd.DataFrame, optional
        Input DataFrame containing laureates' data (default is `df_laureates`).
    year : str or int, optional
        Award year to filter by; use "last" for the most recent year, or 'all'. Individual years won't work. (default is "last").
    category : str, optional
        Nobel Prize category to filter (e.g., 'physics', default is "all").
    gender : str, optional
        Gender filter ('male', 'female', default is "all").

    Returns:
    -------
    plotly.graph_objs._figure.Figure
        A Plotly mapbox visualization with migration paths, birth, and affiliation markers.
    """
    
    # colors = colorscale_palette

    data = prepare_data_splines(data, year, category, gender)


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
            color="darkgrey",
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
            color="darkgrey",
            opacity=0.9
        ),
        name="Affiliation Cities"
    ))

    # Add migration paths for each laureate with curvature
    colors = [c_brown, c_darkmagenta, c_lightblue, c_orange, c_pink, c_red, c_teal]
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
        title_text="Places of Birth & Affiliation",
        showlegend=False,
        mapbox=dict(
            style="carto-positron",  # Other styles: "streets", "dark", "light", "satellite", etc.
            center=dict(lat=30, lon=0),  # Center the map globally
            zoom=0.8,
        
        ),
        # height=800,
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce the margins
    )

    return fig




##################################################################################################
# Generate and Save Plots
##################################################################################################

if __name__ == "__main__":

    # Save CSV files to disk

    df_laureates_enriched_full_clean.to_csv("df_laureates_enriched_full_clean.csv", sep=';', encoding="UTF-8")
    print("df_laureates_enriched_full_clean.csv has been saved.")

    df_prizes_enriched_full_clean.to_csv("df_prizes_enriched_full_clean.csv", sep=';', encoding="UTF-8")
    print("df_prizes_enriched_full_clean.csv has been saved.")

    df_laureates_enriched_redux_clean.to_csv("df_laureates_enriched_redux_clean.csv", sep=';', encoding="UTF-8")
    print("df_laureates_enriched_redux_clean.csv has been saved.")

    df_prizes_enriched_redux_clean.to_csv("df_prizes_enriched_redux_clean.csv", sep=';', encoding="UTF-8")
    print("df_prizes_enriched_redux_clean.csv has been saved.")


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
        fig_map_cities = generate_map_cities(df_laureates)
        fig_bubbles_population = generate_bubbles_population(df_prizes)
        fig_bar_prizespercountry = generate_bar_prizerpercountry(df_prizes)
        fig_bar_prizespercountry_rs = generate_bar_prizerpercountry(df_prizes, runningsum=True)
        
        # Tab Demography
        fig_surface_prizesforwomen = generate_surface_plot_mw(df_prizes, gender="female")
        fig_surface_prizesformenwomen = generate_surface_plot_mw(df_prizes, gender="all")

        # Tab Time
        fig_histogram_timegap = generate_histogram_timegap(df_prizes)
        fig_scatter_timegaptrend = generate_scatterbox_timegaptrend(df_prizes)
        fig_scatterbox_age = generate_scatterbox_age(df_laureates)
        fig_heatmap_age = generate_heatmap_age(df_laureates)

        # Tab Migration
        fig_parcat_migration_dwp = generate_parcat_migration(df_laureates)
        fig_parcat_migration_bpd = generate_parcat_migration(df_laureates, loc1="BirthCountryNow", loc2="Prize0_Affiliation0_Country", loc3="DeathCountryNow", width=1400, height=1800)
        fig_globe_movement = generate_globe_movemment(df_laureates)

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
 
    # pcp_plots = generate_plots()
    # with open('pcp_plots.pkl', 'wb') as f:
    #     pickle.dump(pcp_plots, f)

    # print("All plots precomputed.")


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