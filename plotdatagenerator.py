##################################################################################################
# Library Imports
##################################################################################################

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import polars as pl
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import config as cf
import theme as th

# Register the unified Nobel-Spektrum Plotly templates (Jewel/light, Neon/dark).
# Figures use template='nbl_light'; dark variant is applied via th.apply_theme when needed.
pio.templates['nbl_light'] = th.build_plotly_template(dark=False)
pio.templates['nbl_dark'] = th.build_plotly_template(dark=True)
pio.templates.default = 'nbl_light'
import networkx as nx
from collections import defaultdict
from fractions import Fraction
import time
try:
    import pygraphviz  # noqa: F401  (optional; graphviz layouts fall back to spring if missing)
except ImportError:
    pygraphviz = None
import sys


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

# Nominations - Using Lazy Evaluation for large files
# The LazyFrame is only materialized when .collect() is called
_lf_nominations = pl.scan_csv('nominations_full.csv', separator=';', encoding='utf8')
_lf_edges = pl.scan_csv('df_edges.csv', separator=';', encoding='utf8')

# Small lookup tables - loaded eagerly (they're small and needed for joins)
df_coordinates = pl.read_csv('countries_with_coordinates.csv', separator=';', encoding='utf8')
df_match_edges_country = pl.read_csv('edges_country_match.csv', separator=';', encoding='utf8')

# Create country mapping for standardization
_country_mapping = dict(zip(
    df_match_edges_country["CountryEdges"],
    df_match_edges_country["CountryRegular"]
))

# Build df_edges with lazy evaluation chain
# All transformations are queued, not executed
_lf_edges_enriched = (
    _lf_edges
    # Standardize country names
    .with_columns([
        pl.col("nominator_country").replace(_country_mapping).alias("nominator_country"),
        pl.col("nominee_country").replace(_country_mapping).alias("nominee_country")
    ])
    # Join coordinates for nominators
    .join(
        df_coordinates.lazy().select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominator_lat"),
            pl.col("Longitude").alias("nominator_lon")
        ]),
        left_on="nominator_country",
        right_on="Country",
        how="left"
    )
    # Join coordinates for nominees
    .join(
        df_coordinates.lazy().select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominee_lat"),
            pl.col("Longitude").alias("nominee_lon")
        ]),
        left_on="nominee_country",
        right_on="Country",
        how="left"
    )
    # Cast ID columns for joins
    .with_columns([
        pl.col("nominator_id").cast(pl.Int64),
        pl.col("nominee_id").cast(pl.Int64)
    ])
)

# Materialize nominations for operations that need the full data
# (e.g., building prize lookup tables, counting columns)
df_nominations = _lf_nominations.collect()

# Build prize lookup tables from materialized nominations
_nominator_prizes = df_nominations.select([
    pl.col("nominator_1_id").cast(pl.Int64).alias("nominator_id"),
    pl.col("nominator_1_awarded_prizes").alias("nominator_prizes")
]).unique(subset=["nominator_id"]).drop_nulls(subset=["nominator_id"])

# For nominees, we need to unpivot since there can be multiple nominees per nomination
_nominee_prizes_list = []
for i in range(1, 16):  # Assuming max 15 nominees
    col_id = f"nominee_{i}_id"
    col_prizes = f"nominee_{i}_awarded_prizes"
    if col_id in df_nominations.columns and col_prizes in df_nominations.columns:
        _nominee_prizes_list.append(
            df_nominations.select([
                pl.col(col_id).cast(pl.Int64).alias("nominee_id"),
                pl.col(col_prizes).alias("nominee_prizes")
            ]).drop_nulls(subset=["nominee_id"])
        )

_nominee_prizes = pl.concat(_nominee_prizes_list).unique(subset=["nominee_id"])

# Complete the edges LazyFrame with prize joins and materialize
df_edges = (
    _lf_edges_enriched
    .join(
        _nominator_prizes.lazy(),
        on="nominator_id",
        how="left"
    )
    .join(
        _nominee_prizes.lazy(),
        on="nominee_id", 
        how="left"
    )
    .collect()  # Materialize here - all operations run in one optimized pass
)

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



##################################################################################################
# Helper Functions
##################################################################################################

def min_max_normalize(series):
    """
    Apply Min-Max normalization to scale values to range [0, 1].
    
    Used primarily to normalize population values for bubble size scaling
    in visualizations.
    
    Args:
        series (pd.Series | np.ndarray): Numeric series to normalize.
        
    Returns:
        pd.Series | np.ndarray: Normalized values in range [0, 1].
        
    Example:
        >>> data = pd.Series([10, 20, 30, 40, 50])
        >>> min_max_normalize(data)
        0    0.00
        1    0.25
        2    0.50
        3    0.75
        4    1.00
    """
    return (series - series.min()) / (series.max() - series.min())



##################################################################################################
# Data Cleaning
##################################################################################################

# update the csv/df from the Nobel API with corrected information that has been compiled manually

df_laureates_import.update(df_laureates_corrections)


# There are some non-conventional country names (reference: geonames.org) in the data. These will be replaced now.

def replace_values_in_columns(df, columns, replacement_dict):
    """
    Replace values in specified DataFrame columns using a mapping dictionary.
    
    Used for standardizing country names and other categorical values
    across the dataset (e.g., 'Czech Republic' -> 'Czechia').
    
    Args:
        df (pd.DataFrame): The input DataFrame to modify.
        columns (list[str]): Column names where replacements should be applied.
        replacement_dict (dict[str, str]): Mapping of old values to new values.
        
    Returns:
        pd.DataFrame: DataFrame with replaced values (modified in place).
        
    Example:
        >>> replacements = {'USA': 'United States', 'UK': 'United Kingdom'}
        >>> df = replace_values_in_columns(df, ['BirthCountry', 'DeathCountry'], replacements)
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

# Replace "Physiology or Medicine" with "Medicine" for consistency
df_laureates['Prize0_Category'] = df_laureates['Prize0_Category'].replace('Physiology or Medicine', 'Medicine')
df_laureates['Prize1_Category'] = df_laureates['Prize1_Category'].replace('Physiology or Medicine', 'Medicine')
df_laureates['Prize2_Category'] = df_laureates['Prize2_Category'].replace('Physiology or Medicine', 'Medicine')

df_nominations = df_nominations.with_columns(
    pl.col('nomination_category_from_title').str.replace('Physiology or Medicine', 'Medicine')
)

df_edges = df_edges.with_columns(
    pl.col('category').str.replace('Physiology or Medicine', 'Medicine')
)


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
df_prizes_enriched_redux_clean = df_prizes.copy()


##################################################################################################
# Create Polars dataframes with Lazy capabilities
##################################################################################################

# Convert Pandas to Polars DataFrames
# These are used for the list generator and extended filtering
pldf_laureates_enriched_redux_clean = pl.DataFrame(df_laureates_enriched_redux_clean)
pldf_prizes_enriched_redux_clean = pl.DataFrame(df_prizes_enriched_redux_clean)

# Create LazyFrames for efficient query building
# Use these for complex filter chains - they'll be optimized before execution
lf_laureates = pldf_laureates_enriched_redux_clean.lazy()
lf_prizes = pldf_prizes_enriched_redux_clean.lazy()

##################################################################################################
# Functions for Prize Statistics
##################################################################################################

def get_lastyearincluded(data=df_prizes):
    """
    Get the most recent award year from the prizes dataset.
    
    Used to dynamically set time range boundaries in filters and plots.
    
    Args:
        data (pd.DataFrame): DataFrame containing prize data with 'Prize0_AwardYear' column.
            Defaults to df_prizes.
            
    Returns:
        int: The most recent award year in the dataset.
        
    Example:
        >>> lastyear = get_lastyearincluded()
        >>> print(lastyear)  # e.g., 2024
    """
    data = data.sort_values(by="Prize0_AwardYear", ascending=True)
    lastyearincluded = data.iloc[-1]["Prize0_AwardYear"]
    return lastyearincluded

lastyearincluded = get_lastyearincluded(df_prizes)

def get_currentlaureatemotivations(data=df_prizes):

    # Filter df for current year
    df_currentprizes = data[data["Prize0_AwardYear"] == lastyearincluded]
    
    # Write different categories to separate dfs
    df_currentprizes_med = df_currentprizes[df_currentprizes["Prize0_Category"] == "Medicine"].sort_index()
    df_currentprizes_phys = df_currentprizes[df_currentprizes["Prize0_Category"] == "Physics"].sort_index()
    df_currentprizes_chem = df_currentprizes[df_currentprizes["Prize0_Category"] == "Chemistry"].sort_index()
    df_currentprizes_lit = df_currentprizes[df_currentprizes["Prize0_Category"] == "Literature"].sort_index()
    df_currentprizes_peace = df_currentprizes[df_currentprizes["Prize0_Category"] == "Peace"].sort_index()
    df_currentprizes_eco = df_currentprizes[df_currentprizes["Prize0_Category"] == "Economic Sciences"].sort_index()

    # Turn the dfs into dicts
    dict_med = df_currentprizes_med.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()
    dict_phys = df_currentprizes_phys.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()
    dict_chem = df_currentprizes_chem.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()
    dict_lit = df_currentprizes_lit.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()
    dict_peace = df_currentprizes_peace.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()
    dict_eco = df_currentprizes_eco.set_index("AwardeeDisplayName")["Prize0_Motivation"].to_dict()


    def generate_html_current_prizes(prize_dict):

        # check for repeating motivations, and keep only the last
        items = list(prize_dict.items())
        new_items = []
        for i in range(len(items)):
            key, value = items[i]
            # Check if there is a next item and if its value equals the current value.
            if i < len(items) - 1:
                _, next_value = items[i + 1]
                if value == next_value:
                    # Replace the first of the identical pair with an empty string (or "skip" if desired)
                    new_items.append((key, ''))
                else:
                    new_items.append((key, value))
            else:
                # Always keep the last item as is
                new_items.append((key, value))

        data = dict(new_items)


        # turn the dict to a list of html elements
        html_elements = []
        for key, value in data.items():
            # Create and append the <h4> element for the key.
            html_elements.append(f"<h4>{key}</h4>")
            # If the value is non-empty, create and append the <p> element.
            if value:
                html_elements.append(f"<p>{value}</p>")

        # If you want to see the result as a single string:
        #html_output = "\n".join(html_elements)
        html_output = "".join(html_elements)
        # print(html_output)

        return html_output
    
    html_med = generate_html_current_prizes(dict_med)
    html_phys = generate_html_current_prizes(dict_phys)
    html_chem = generate_html_current_prizes(dict_chem)
    html_lit = generate_html_current_prizes(dict_lit)
    html_peace = generate_html_current_prizes(dict_peace)
    html_eco = generate_html_current_prizes(dict_eco)

    return html_med, html_phys, html_chem, html_lit, html_peace, html_eco
    




##################################################################################################
# Count Laureates per Country
##################################################################################################

def count_per_country(data=df_laureates, country="BirthCountryNow"):
    """
    Count Nobel laureates per country with ISO3 codes for mapping.
    
    Aggregates laureate counts by country and enriches the result
    with ISO3 country codes needed for choropleth visualizations.
    
    Args:
        data (pd.DataFrame): DataFrame containing laureate data.
            Defaults to df_laureates.
        country (str): Column name to count by. Common options:
            - 'BirthCountryNow': Country of birth
            - 'Prize0_Affiliation0_CountryNow': Affiliation country at time of award
            - 'DeathCountryNow': Country of death
            Defaults to 'BirthCountryNow'.
            
    Returns:
        pd.DataFrame: DataFrame with columns ['Country', 'Count', 'ISO2', 'ISO3'].
        
    Example:
        >>> df_counts = count_per_country()
        >>> df_counts.head()
           Country  Count ISO2  ISO3
        0      USA    403   US   USA
        1       UK     98   GB   GBR
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
df_max_prize_count=count_per_country()
max_prize_count = df_max_prize_count['Count'].max()


##################################################################################################
# Standard Filter
##################################################################################################

def define_category_states(chip_medicine, chip_physics, chip_chemistry, chip_economics, chip_literature, chip_peace):
    """
    Convert individual category chip states to a list of selected categories.
    
    Maps the boolean state of UI filter chips to a list of category names
    for use in data filtering operations.
    
    Args:
        chip_medicine (bool): Whether Medicine category is selected.
        chip_physics (bool): Whether Physics category is selected.
        chip_chemistry (bool): Whether Chemistry category is selected.
        chip_economics (bool): Whether Economic Sciences category is selected.
        chip_literature (bool): Whether Literature category is selected.
        chip_peace (bool): Whether Peace category is selected.
        
    Returns:
        list[str]: List of selected category names (e.g., ['Physics', 'Chemistry']).
        
    Example:
        >>> categories = define_category_states(True, True, False, False, False, False)
        >>> print(categories)
        ['Medicine', 'Physics']
    """
    # Map chip states to categories
    chip_states = {
        "Medicine": chip_medicine,
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

def define_gender_states(chip_female, chip_male):
    """
    Convert gender chip states to a filter string.
    
    Maps the boolean state of gender filter chips to a string value
    for use in data filtering operations.
    
    Args:
        chip_female (bool): Whether female filter is selected.
        chip_male (bool): Whether male filter is selected.
        
    Returns:
        str: Filter value - 'all', 'male', 'female', or '' (empty if none selected).
        
    Example:
        >>> gender = define_gender_states(True, False)
        >>> print(gender)
        'female'
    """

    # Determine the gender filter based on the chip states
    if chip_female and chip_male:
        return "all"
    elif chip_male:
        return "male"
    elif chip_female:
        return "female"
    else:
        return ""


def define_type_states(chip_humans, chip_organizations):
    """
    Convert laureate type chip states to a filter string.
    
    Maps the boolean state of type filter chips (humans vs organizations)
    to a string value for use in data filtering operations.
    
    Args:
        chip_humans (bool): Whether human laureates filter is selected.
        chip_organizations (bool): Whether organization laureates filter is selected.
        
    Returns:
        str: Filter value - 'all', 'human', 'organization', or '' (empty if none selected).
        
    Example:
        >>> laureate_type = define_type_states(True, False)
        >>> print(laureate_type)
        'human'
    """
    # Determine the type filter based on the chip states
    if chip_humans and chip_organizations:
        return "all"
    elif chip_humans:
        return "human"
    elif chip_organizations:
        return "organization"
    else:
        return ""
    
def define_alive_states(chip_alive, chip_dead):
    """
    Convert alive/dead chip states to a filter string.
    
    Maps the boolean state of alive/dead filter chips to a string value
    for use in data filtering operations.
    
    Args:
        chip_alive (bool): Whether living laureates filter is selected.
        chip_dead (bool): Whether deceased laureates filter is selected.
        
    Returns:
        str: Filter value - 'all', 'alive', 'dead', or '' (empty if none selected).
        
    Example:
        >>> status = define_alive_states(True, False)
        >>> print(status)
        'alive'
    """
    # Determine the alive/dead filter based on the chip states
    if chip_alive and chip_dead:
        return "all"
    elif chip_alive:
        return "alive"
    elif chip_dead:
        return "dead"
    else:
        return ""   


def standard_filter(data, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", callsign=""):
    """
    Apply standard filters to Nobel laureate/prize data.
    
    Central filtering function used by most plot generators to filter data
    by category, gender, and time range.
    
    Args:
        data (pd.DataFrame): Input DataFrame (typically df_laureates or df_prizes).
        categories (str | list[str]): Categories to include. Options:
            - 'all': All six categories
            - 'sci': Science categories (Medicine, Physics, Chemistry, Economics)
            - 'natsci': Natural sciences (Medicine, Physics, Chemistry)
            - list: Specific category names
            Defaults to 'all'.
        gender (str): Gender filter - 'all', 'male', or 'female'. Defaults to 'all'.
        timerange (list[int]): [start_year, end_year] for filtering. 
            Defaults to [1901, lastyearincluded].
        timerange_field (str): Which date field to filter on:
            - 'award': Prize0_AwardYear
            - 'birth': BirthDate (year extracted)
            - 'death': DeathDate (year extracted)
            Defaults to 'award'.
        callsign (str): Optional identifier for debugging. Defaults to ''.
        
    Returns:
        pd.DataFrame: Filtered DataFrame matching all specified criteria.
        
    Example:
        >>> filtered = standard_filter(df_laureates, categories='natsci', gender='female')
        >>> filtered = standard_filter(df_prizes, timerange=[1950, 2000])
    """

    # Replace short handles with lists
    if categories == "all":
        categories=["Medicine", "Physics", "Chemistry", "Economic Sciences", "Literature", "Peace"]
    elif categories == "sci":
        categories=["Medicine", "Physics", "Chemistry", "Economic Sciences"]
    elif categories == "natsci":
        categories=["Medicine", "Physics", "Chemistry"]
    else:
        categories = categories

    # print(f"Categories from: {callsign}: {categories}")

    # Apply category filter
    df_filtered = data[data["Prize0_Category"].isin(categories)]

    # Apply gender filter if not "all"
    if gender.lower() != "all":
        df_filtered = df_filtered[df_filtered["LaureateGender"] == gender]
    else:
        df_filtered = df_filtered


    # print(f"Time Range: {timerange}")
    # print(f"Time Range Field: {timerange_field}")
    #print("Columns in DataFrame:", df_filtered.columns)

    if timerange is None:
        df_filtered = df_filtered
    else:
        if timerange_field == "birth":
            df_filtered['BirthYear'] = pd.to_datetime(df_filtered['BirthDate'], errors='coerce', format='%Y-%m-%d').dt.year
            df_filtered['BirthYear'] = df_filtered['BirthYear'].fillna(0).astype(int)
            datefield = 'BirthYear'
        elif timerange_field == "death":
            df_filtered['DeathYear'] = pd.to_datetime(df_filtered['DeathDate'], errors='coerce', format='%Y-%m-%d').dt.year
            df_filtered['DeathYear'] = df_filtered['DeathYear'].fillna(0).astype(int)
            datefield = 'DeathYear'
        elif timerange_field == "award":
            df_filtered.loc[:,'Prize0_AwardYear'] = df_filtered['Prize0_AwardYear'].fillna(0).astype(int)
            datefield = 'Prize0_AwardYear'
        else:
            df_filtered.loc[:,'Prize0_AwardYear'] = df_filtered['Prize0_AwardYear'].fillna(0).astype(int)
            datefield = 'Prize0_AwardYear'

        # print("Datefield:", datefield)
        # print("Timerange:", timerange)
        # print("Columns in DataFrame:", df_filtered.columns)
        # print("DataFrame Object Type:", df_filtered.dtypes)

        # print(f"Datefield dtype: {df_filtered[datefield].dtype}")
        # print(f"Timerange dtype: {type(timerange[0])}")

        df_filtered = df_filtered[
            (df_filtered[datefield].astype(int) >= timerange[0]) &
            (df_filtered[datefield].astype(int) <= timerange[1])
        ]

    # print(f"Result rows/columns: {df_filtered.shape[0]}, {df_filtered.shape[1]}")

    return df_filtered


def replace_country_designations(country):
    """
    Convert short country type identifiers to actual column names.
    
    Maps user-friendly country type strings to the corresponding
    DataFrame column names.
    
    Args:
        country (str): Short identifier - 'birth', 'affiliation', or 'death'.
        
    Returns:
        str: Corresponding column name in the DataFrame.
        
    Example:
        >>> col_name = replace_country_designations('birth')
        >>> print(col_name)
        'BirthCountryNow'
    """
    country = country.replace("birth", "BirthCountryNow")
    country = country.replace("affiliation", "Prize0_Affiliation0_CountryNow")
    country = country.replace("death", "DeathCountryNow")
    return country


##################################################################################################
# Extended Filter (POLARS) - with Lazy Evaluation
##################################################################################################

def extended_filter(data=pldf_laureates_enriched_redux_clean, categories="all", gender="all", type="all", alive="all", numberofprizes=1, countries_of_birth="all", countries_of_affiliation="all", timerange_birth=[1817, lastyearincluded-25], timerange_award=[1901, lastyearincluded], motivation_input="", search_mode="all", output_options="compact", callsign=""):
    """
    Apply extended filters to laureate data using Polars with Lazy Evaluation.
    
    Builds a query plan that is optimized and executed only when results are needed.
    Supports complex filtering by category, gender, type, life status, countries,
    time ranges, and free-text motivation search.
    
    Args:
        data (pl.DataFrame): Input Polars DataFrame. Defaults to pldf_laureates_enriched_redux_clean.
        categories (str | list): Category filter - 'all', 'sci', 'natsci', or list.
        gender (str): 'all', 'male', or 'female'.
        type (str): 'all', 'human', or 'organization'.
        alive (str): 'all', 'alive', or 'dead'.
        numberofprizes (int): Minimum number of prizes (1, 2, or 3).
        countries_of_birth (str | list): 'all' or list of country names.
        countries_of_affiliation (str | list): 'all' or list of country names.
        timerange_birth (list): [start_year, end_year] for birth date filter.
        timerange_award (list): [start_year, end_year] for award date filter.
        motivation_input (str): Free-text search in motivation field.
        search_mode (str): 'all' (AND) or 'any' (OR) for multi-term search.
        output_options (str): 'names', 'compact', 'extended', or 'full'.
        callsign (str): Debug identifier.
        
    Returns:
        pl.DataFrame: Filtered and formatted DataFrame.
        
    Performance:
        Uses lazy evaluation - all filter operations are queued and optimized
        before execution. The query plan is executed with .collect() at the end.
    """
    
    # Replace short for categories handles with lists
    if categories == "all":
        categories=["Medicine", "Physics", "Chemistry", "Economic Sciences", "Literature", "Peace"]
    elif categories == "sci":
        categories=["Medicine", "Physics", "Chemistry", "Economic Sciences"]
    elif categories == "natsci":
        categories=["Medicine", "Physics", "Chemistry"]
    else:
        pass

        
    # Replace short handles for countries_of_birth with lists
    if countries_of_birth in ["all", None, []]:
        countries_of_birth = countries_to_list(data)
        countries_of_birth.append("")
    else:
        pass
    
    
    # Replace short handles for countries_of_affiliation with lists
    if countries_of_affiliation in ["all", None, []]:
        countries_of_affiliation = countries_to_list(data, column="Prize0_Affiliation0_CountryNow")
    else:
        pass
   

    def is_not_empty(column_name):
        """Check if column is not empty (String or INT)"""
        return (
            ~pl.col(column_name).is_null() &
            ~pl.col(column_name).cast(pl.Utf8).str.strip_chars().is_in(["", "None", "NaN", "null"])
        )

    # Convert to LazyFrame for optimized query building
    # All operations below are queued, not executed
    lf = data.lazy()
    
    ### FILTER: CATEGORIES ###
    lf = lf.filter(pl.col("Prize0_Category").is_in(categories))

    ### FILTER: GENDER ###
    if gender.lower() != "all":
        lf = lf.filter(pl.col("LaureateGender") == gender)

    ### FILTER: TYPE ###
    if type.lower() == "organization":
        lf = lf.filter(is_not_empty("OrganisationName"))
    elif type.lower() == "human":
        lf = lf.filter(is_not_empty("LaureateNameLast"))
    elif type.lower() == "":
        lf = lf.filter(~is_not_empty("AwardeeDisplayName"))

    ### FILTER: ALIVE/DEAD ###
    if alive.lower() == "alive":
        lf = lf.filter(~is_not_empty("DeathDate"))
    elif alive.lower() == "dead":
        lf = lf.filter(is_not_empty("DeathDate"))
    elif alive.lower() == "":
        lf = lf.filter(~is_not_empty("AwardeeDisplayName"))

    ### FILTER: NUMBER OF PRIZES ###  
    if numberofprizes == 2:
        lf = lf.filter(is_not_empty("Prize1_AwardYear"))
    elif numberofprizes == 3:
        lf = lf.filter(is_not_empty("Prize2_AwardYear"))

    ### FILTER: COUNTRIES OF BIRTH ###
    lf = lf.filter(
        pl.col("BirthCountryNow").is_in(countries_of_birth) | 
        pl.col("BirthCountryNow").is_null() | 
        (pl.col("BirthCountryNow") == "") | 
        (pl.col("BirthCountryNow") == "None")
    )

    ### FILTER: COUNTRIES OF AFFILIATION ###
    lf = lf.filter(
        pl.col("Prize0_Affiliation0_CountryNow").is_in(countries_of_affiliation) | 
        pl.col("Prize0_Affiliation0_CountryNow").is_null() | 
        (pl.col("Prize0_Affiliation0_CountryNow") == "") | 
        (pl.col("Prize0_Affiliation0_CountryNow") == "None")
    )
          
    ### FILTER: BIRTH YEAR ###
    if timerange_birth is not None:
        lf = lf.with_columns([
            pl.col("BirthDate")
            .str.to_datetime(format="%Y-%m-%d", strict=False)
            .dt.year()
            .fill_null(0)
            .cast(pl.Int32)
            .alias("BirthYear")
        ])
        
        lf = lf.filter(
            pl.col("BirthYear").is_between(timerange_birth[0], timerange_birth[1]) | 
            (pl.col("BirthYear") == 0)
        )

    ### FILTER: AWARD YEAR ###
    if timerange_award is not None:  
        lf = lf.with_columns([
            pl.col("Prize0_AwardYear")
            .cast(pl.Int32, strict=False)
            .fill_null(0)
        ])
               
        lf = lf.filter(
            pl.col("Prize0_AwardYear").is_between(timerange_award[0], timerange_award[1]) | 
            (pl.col("Prize0_AwardYear") == 0)
        )
    # print(f"GrÃƒÂ¶ÃƒÅ¸e Datensatz nachher: {df_filtered.shape}")


    ### FILTER: MOTIVATION ###

    columns = ['Prize0_Motivation', 'Prize1_Motivation', 'Prize2_Motivation']

    def search_in_columns_simple(search_terms, columns, mode="any"):
        
        # concatenate all columns into a single string column
        combined_text = pl.concat_str([
            pl.col(col).cast(pl.Utf8).fill_null("") for col in columns
        ], separator=" ")
        
        if mode.lower() == "any":
            # at least one term must be present
            conditions = [
                combined_text.str.contains(f"(?i){term}") 
                for term in search_terms
            ]
            return pl.any_horizontal(conditions)
        
        elif mode.lower() == "all":
            # all terms must be present
            conditions = [
                combined_text.str.contains(f"(?i){term}") 
                for term in search_terms
            ]
            return pl.all_horizontal(conditions)

    # Only filter if there is a motivation input
    if motivation_input and len(motivation_input) > 0:
        lf = lf.filter(search_in_columns_simple(motivation_input, columns, mode=search_mode))

    
    ### OUTPUT OPTIONS ###
    # For output options that need to check data presence, we need to collect first
    # This is a tradeoff - we collect once and then do the final selection
    
    if output_options == "names":
        # Select only the names of the laureates
        lf = lf.select([
            pl.col("AwardeeDisplayName").alias("Name"),
        ])

    elif output_options == "compact":
        # Select a compact set of columns
        lf = lf.select([
            pl.col("AwardeeDisplayName").alias("Name"),
            pl.col("Prize0_AwardYear").alias("Award Year"),
            pl.col("Prize0_Category").alias("Category"),
            pl.col("Prize0_Motivation").alias("Motivation"),
            pl.col("BirthCountryNow").alias("Birth Country"),
            pl.col("Prize0_Affiliation0_NameNow").alias("Affiliation at Time of Award"),
            pl.col("Prize0_Affiliation0_CountryNow").alias("Affiliation Country")
        ])

    elif output_options == "extended":
        # For extended output, we need to collect to check for Prize1/Prize2 presence
        # This is unavoidable as we need to inspect the data
        df_collected = lf.collect()
        
        # base columns for the extended output
        base_columns = [
            pl.col("AwardeeDisplayName").alias("Name"),
            pl.col("LaureateNameLast").alias("Last Name"),
            pl.col("LaureateNameFirst").alias("First Name"),
            pl.col("OrganisationName").alias("Organisation Name"),
            pl.col("LaureateGender").alias("Gender"),
            pl.col("BirthCountryNow").alias("Birth Country"),
            pl.col("BirthDate").alias("Birth Date"),
            pl.col("DeathCountryNow").alias("Death Country"),
            pl.col("DeathDate").alias("Death Date"),
            pl.col("Prize0_AwardYear").alias("Prize Award Year"),
            pl.col("Prize0_Category").alias("Prize Category"),
            pl.col("Prize0_Motivation").alias("Prize Motivation"),
            pl.col("Prize0_Affiliation0_NameNow").alias("Affiliation at Time of Award"),
            pl.col("Prize0_Affiliation0_CityNow").alias("Affiliation City at Time of Award"),
            pl.col("Prize0_Affiliation0_CountryNow").alias("Affiliation Country at Time of Award")
        ]
        
        # if there is a second prize, add the columns for the second prize
        has_prize1 = df_collected.filter(~pl.col("Prize1_AwardYear").is_null()).height > 0
        if has_prize1:
            base_columns.extend([
                pl.col("Prize1_AwardYear").alias("Second Prize Award Year"),
                pl.col("Prize1_Category").alias("Second Prize Category"),
                pl.col("Prize1_Motivation").alias("Second Prize Motivation"),
                pl.col("Prize1_Affiliation0_NameNow").alias("Second Prize Affiliation at Time of Award"),
                pl.col("Prize1_Affiliation0_CityNow").alias("Second Prize Affiliation City at Time of Award"),
                pl.col("Prize1_Affiliation0_CountryNow").alias("Second Prize Affiliation Country at Time of Award")
            ])
        
        # Check if Prize2 data exists
        has_prize2 = df_collected.filter(~pl.col("Prize2_AwardYear").is_null()).height > 0
        if has_prize2:
            base_columns.extend([
                pl.col("Prize2_AwardYear").alias("Third Prize Award Year"),
                pl.col("Prize2_Category").alias("Third Prize Category"),
                pl.col("Prize2_Motivation").alias("Third Prize Motivation"),
            ])
        
        # Return directly with selection applied
        return df_collected.select(base_columns)

    elif output_options == "full":
        pass

    else:
        raise ValueError(f"Invalid output option: {output_options}")

    # Execute the lazy query plan and return results
    # All filter operations are optimized and run in a single pass
    return lf.collect()


##################################################################################################
# List of Countries (POLARS)
##################################################################################################

def countries_to_list(data=pldf_laureates_enriched_redux_clean, column="BirthCountryNow"):
    """
   Extract alphabetically sorted list of unique countries from a Polars DataFrame.
   
   Args:
       data (pl.DataFrame): Polars DataFrame containing Nobel laureate data.
                          Default: pldf_laureates_enriched_redux_clean
       column (str): Column name to extract countries from. Default: "BirthCountryNow"
   
   Returns:
       list[str]: Alphabetically sorted list of unique country names, excluding "None" values.
       
   Example:
       >>> countries = countries_to_list()
       >>> print(countries[:3])
       ['Austria', 'Belgium', 'Canada']
       
       >>> death_countries = countries_to_list(column="DeathCountryNow")
   """
    pldf_result = data.select(
        pl.col(column)
        .filter(pl.col(column) != "None")
        .unique()
        .sort()).to_series().to_list()
    return pldf_result


##################################################################################################
# Plots
##################################################################################################

# Geography
##################################################################################################

# Plot: Choropleth Globe Countries
# ================================================================================================

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
            color = cf.c_brand_color_main,
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
            # bgcolor='#ffffff',
            # landcolor='#f0f0f0',
            # showcountries=True,
            # oceancolor=' #f0f005',
            # rivercolor=' #f0f0f5',
            # lakecolor=' #f0f0f5',
            # showframe=False,
            # showcoastlines=False,
            # projection_type='orthographic'

            projection_type="orthographic",
            showland=True,
            countrycolor=cf.c_black,  # Darker color for country borders
            countrywidth=0.8,  # Border width
            coastlinecolor=cf.c_black,  # Darker coastlines
            coastlinewidth=0.5,  # Coastline width
            showlakes=True,
            showcountries=True,
            showocean=True,
            showframe=False,  # Removes the box frame
            bgcolor='#ffffff',
            landcolor='#f0f0f0',
            oceancolor=' #e6f2ff',
            rivercolor=' #e6f2ff',
            lakecolor=' #e6f2ff',

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

    # Hover label styling
    fig.update_layout(
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )

    return fig


# Plot: Scatter Mapbox Cities
# ================================================================================================

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
            color=cf.c_blue_light,  # Color of the marker (your brand color)
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
        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 14,
            color = cf.c_brand_color_main
        ),
        hoverlabel=dict(
                bgcolor=cf.c_hoverlabel_bg,
                font_size=12,
                font_family="IBM Plex Sans"
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
        height = 800,
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


# generate the plot

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
                color = cf.c_brand_color_main,
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


# Plot: Prizes per Country Stacked Bar Chart
# ================================================================================================

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
            color=cf.c_brand_color_main,
        ),
        hoverlabel=dict(
            bgcolor=cf.c_plot_background,
            font_size=12,
            font_family="IBM Plex Sans"
        )
    )


    return fig


# Prizes per Country Stacked Bar Chart (Running Sum)
# ================================================================================================

# fig=generate_bar_percountry(runningsum=True)



# Demography
##################################################################################################



# Plot: Prizes to Women and Men per Decade, 3D surface
# ================================================================================================

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


# Plot: Donuts Gender/Ethnicity/Religion
# ================================================================================================

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



# Series: Time
##################################################################################################


# Plot: Timegap
# ================================================================================================

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
            color = cf.c_brand_color_main,
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



# Plot: Timegap with Trendlines
# ================================================================================================

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
        line=dict(color=cf.c_black, width=2),  # Red line for the new data
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
        line=dict(color=cf.c_blue_light, width=2),  # Red line for the new data
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
            color = cf.c_brand_color_main,
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


# Plot: Age Scatterplot
# ================================================================================================


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
            color = cf.c_brand_color_main,
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



# Plot: Age Heatmap
# ================================================================================================

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
            color = cf.c_brand_color_main,
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


# Migration
##################################################################################################

# Plot: Movement Degree - Work - Prize
# ================================================================================================

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
            color = cf.c_brand_color_main,
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



# Misc
##################################################################################################

def get_conversion_rates():
    """
    Returns conversion rates for SEK to EUR and SEK to USD.
    
    Uses static rates from config.py instead of fetching from yfinance
    to avoid API rate limits and network dependencies.
    
    Returns:
        tuple: (SEK-to-EUR rate, SEK-to-USD rate)
    """
    return cf.SEK_TO_EUR_RATE, cf.SEK_TO_USD_RATE
    

def generate_line_prizemoney(data=df_prizes, categories="all", gender="all", timerange=[1901, lastyearincluded], timerange_field="award", currency="EUR"):
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
    
    data = standard_filter(data, categories, gender, timerange, timerange_field)


    # Create a deep copy of the relevant columns to avoid the SettingWithCopyWarning
    df_prizemoney = data[["Prize0_AwardYear", "Prize0_Category", "Prize0_Portion", "Prize0_Amount", "Prize0_AmountAdjusted_"]].copy(deep=True)

    # Convert Prize0_Portion ("1", "1/2", "1/4", ...) to numeric (direct assignment
    # replaces the column dtype; newer pandas rejects in-place dtype changes via .loc)
    df_prizemoney["Prize0_Portion"] = df_prizemoney["Prize0_Portion"].apply(lambda x: float(Fraction(x)))

    # Calculate PrizeAmountShared
    df_prizemoney["PrizeAmountShared"] = df_prizemoney["Prize0_Amount"] * df_prizemoney["Prize0_Portion"] * conversionrate

    # Calculate PrizeAmountAdjustedShared
    df_prizemoney["PrizeAmountAdjustedShared"] = df_prizemoney["Prize0_AmountAdjusted_"] * df_prizemoney["Prize0_Portion"]  * conversionrate

    df_prizemoney_rs = df_prizemoney.groupby("Prize0_AwardYear").agg({
        "PrizeAmountShared": "sum",
        "PrizeAmountAdjustedShared": "sum"
    }).reset_index()

    # Step 2: Calculate the running sum for each column
    df_prizemoney_rs["CumulativePrizeAmountShared"] = df_prizemoney_rs["PrizeAmountShared"].cumsum()
    df_prizemoney_rs["CumulativePrizeAmountAdjustedShared"] = df_prizemoney_rs["PrizeAmountAdjustedShared"].cumsum()

    data = df_prizemoney_rs

    # Create a line chart with Plotly
    fig = go.Figure()

    # Add the line for the running sum
    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountShared"],
        mode='lines+markers',
        name=f'Cumulative Prize Amount ({currencyname})',
        line=dict(color=cf.c_brand_color_alt),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=data["Prize0_AwardYear"],
        y=data["CumulativePrizeAmountAdjustedShared"],
        mode='lines+markers',
        name=f'Cumulative Prize Amount ({currencyname}) Inflation Adjusted',
        line=dict(color=cf.c_brand_color_acc),
        marker=dict(size=8)
    ))
    # Customize layout
    fig.update_layout(

        xaxis_title="Year",
        yaxis_title=f"Cumulative Prize Amount ({currencyname})",
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,

        margin={"r":0,"t":60,"l":0,"b":0},

        font=dict(
            family = 'IBM Plex Sans, sans-serif',
            size = 11,
            color = cf.c_brand_color_main,
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

    fig.update_traces(
        hovertemplate=(
            "<b>Year:</b> %{x}<br>" +
            "Amount: %{y}<br>"
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

def generate_var_prizeamount(data=df_prizes, currency="EUR"):
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

    # Convert Prize0_Portion ("1", "1/2", "1/4", ...) to numeric (direct assignment
    # replaces the column dtype; newer pandas rejects in-place dtype changes via .loc)
    df_prizemoney["Prize0_Portion"] = df_prizemoney["Prize0_Portion"].apply(lambda x: float(Fraction(x)))

    # Calculate PrizeAmountShared
    df_prizemoney["PrizeAmountShared"] = df_prizemoney["Prize0_Amount"] * df_prizemoney["Prize0_Portion"] * conversionrate

    # Calculate PrizeAmountAdjustedShared
    df_prizemoney["PrizeAmountAdjustedShared"] = df_prizemoney["Prize0_AmountAdjusted_"] * df_prizemoney["Prize0_Portion"]  * conversionrate

    df_prizemoney_rs = df_prizemoney.groupby("Prize0_AwardYear").agg({
        "PrizeAmountShared": "sum",
        "PrizeAmountAdjustedShared": "sum"
    }).reset_index()

    # Step 2: Calculate the running sum for each column
    df_prizemoney_rs["CumulativePrizeAmountShared"] = df_prizemoney_rs["PrizeAmountShared"].cumsum()
    df_prizemoney_rs["CumulativePrizeAmountAdjustedShared"] = df_prizemoney_rs["PrizeAmountAdjustedShared"].cumsum()

    data = df_prizemoney_rs
    totalprizeamount = data["CumulativePrizeAmountAdjustedShared"].iloc[-1]

    return totalprizeamount

totalprizeamount = generate_var_prizeamount(data=df_prizes, currency="EUR")


# Current Facts
##################################################################################################

# Plot Sunburst Current
#=================================================================================================

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



# Plot Movement Splines on Map/Globe
#=================================================================================================

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
            color=cf.c_black,
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
            countrycolor=cf.c_black,  # Darker color for country borders
            countrywidth=0.8,  # Border width
            coastlinecolor=cf.c_black,  # Darker coastlines
            coastlinewidth=0.5,  # Coastline width
            showlakes=True,
            showcountries=True,
            showocean=True,
            showframe=False,  # Removes the box frame
            bgcolor='#ffffff',
            landcolor='#f0f0f0',
            oceancolor='#f1f6ff',
            rivercolor=' #e6f2ff',
            lakecolor=' #e6f2ff',


        ),
        height=900,
        #autosize=True
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
            color=cf.c_teal,
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
            color=cf.c_teal,
            opacity=0.9
        ),
        name="Affiliation Cities"
    ))

    # Add migration paths for each laureate with curvature
    # colors = [cf.c_brown, cf.c_darkmagenta, cf.c_lightblue, cf.c_orange, cf.c_pink, cf.c_red, cf.c_teal]
    colors = cf.c_colorscale_palette
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
            style="carto-positron",  # Other styles: "streets", "dark", "light", "satellite", etc.
            center=dict(lat=30, lon=0),  # Center the map globally
            zoom=0.8,
        
        ),

        hoverlabel=dict(
                bgcolor=cf.c_hoverlabel_bg,
                font_size=12,
                font_family="IBM Plex Sans"
        ),
        # height=800,
        margin=dict(l=0, r=0, t=0, b=0),  # Reduce the margins
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

    bar_colors = [
        cf.c_black_light,
        cf.c_blue_light,
        cf.c_teal_light,
        cf.c_green_light,
        cf.c_yellow_light,
        cf.c_orange_light,
        cf.c_red_light,
        cf.c_purple_light,
        cf.c_grey_light,
        cf.c_black,
        cf.c_blue,
        cf.c_teal,
        cf.c_green,
        cf.c_yellow,
        cf.c_orange,
        cf.c_red,
        cf.c_purple,
        cf.c_grey,
        cf.c_black_dark,
        cf.c_blue_dark,
        cf.c_teal_dark,
        cf.c_green_dark,
        cf.c_yellow_dark,
        cf.c_orange_dark,
        cf.c_red_dark
    ][:len(top_names)]

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

            font=dict(
                family = 'IBM Plex Sans, sans-serif',
                size = 11,
                color = cf.c_brand_color_main,
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


# Nominations
##################################################################################################

# Plot Network Nominations
#=================================================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_max_nominee_count(df):
    """Count how many nominee columns exist"""
    count = 0
    i = 1
    while f'nominee_{i}_name' in df.columns:
        count += 1
        i += 1
    return count

def is_valid_value(value):
    """Check if value is not None and not empty string"""
    return value is not None and value != ''

max_nominees = get_max_nominee_count(df_nominations)


# ============================================================================
# Filter Nominations
# ============================================================================

def filter_edges(data, categories="all", timerange_nomination=[1901, 1902], timerange_field="nomination", nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, nominee_name="", nominee_search_mode="all", nominee_gender="all", nominee_country="all", nominee_islaureate=False, expand_network=True):
    """
    Filter nomination edges with optional ego-network expansion.
    
    When name filters are used and expand_network=True, this function will:
    1. Find all persons matching the name filter (primary nodes)
    2. Expand to include ALL their connections (secondary nodes)
    3. Mark edges with 'is_primary_nominator' and 'is_primary_nominee' flags
    
    This allows showing the complete network around filtered persons,
    not just edges where both parties match the filter.
    
    Args:
        data: Polars DataFrame with edge data
        categories: Category filter
        timerange_nomination: Year range filter
        nominator_name: Search term for nominator names
        nominee_name: Search term for nominee names
        expand_network: If True, expand to show full ego-network of matched persons
        ... (other filters)
    
    Returns:
        Filtered Polars DataFrame with additional columns:
        - is_primary_nominator: True if nominator matched the name filter
        - is_primary_nominee: True if nominee matched the name filter
    """

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
            pl.col("nominator_country").is_in(nominator_country)
        )

    ### FILTER: NOMINEE COUNTRY ###
    if nominee_country and nominee_country != "all" and len(nominee_country) > 0:
        if isinstance(nominee_country, str):
            nominee_country = [nominee_country]
        df_filtered = df_filtered.filter(
            pl.col("nominee_country").is_in(nominee_country)
        )

    # ### FILTER: NOMINATOR COUNTRY ###
    # if nominator_country and nominator_country != "all" and len(nominator_country) > 0:
    #     if isinstance(nominator_country, str):
    #         nominator_country = [nominator_country]
    #     df_filtered = df_filtered.filter(
    #         pl.col("nominator_country").is_in(nominator_country) | 
    #         pl.col("nominator_country").is_null() | 
    #         (pl.col("nominator_country") == "Unknown")
    #     )
    
    # ### FILTER: NOMINEE COUNTRY ###
    # if nominee_country and nominee_country != "all" and len(nominee_country) > 0:
    #     if isinstance(nominee_country, str):
    #         nominee_country = [nominee_country]
    #     df_filtered = df_filtered.filter(
    #         pl.col("nominee_country").is_in(nominee_country) | 
    #         pl.col("nominee_country").is_null() | 
    #         (pl.col("nominee_country") == "Unknown")
    #     )
    
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

    # =========================================================================
    # EGO-NETWORK EXPANSION
    # =========================================================================
    # When name filters are used, we first find matching persons (primary),
    # then expand to include ALL their connections (secondary nodes)
    
    has_name_filter = (nominator_name and len(nominator_name) > 0) or (nominee_name and len(nominee_name) > 0)
    
    if has_name_filter and expand_network:
        # Step 1: Find primary person IDs based on name filters
        primary_nominator_ids = set()
        primary_nominee_ids = set()
        
        if nominator_name and len(nominator_name) > 0:
            columns = ['nominator_name']
            matching_nominators = df_filtered.filter(search_in_columns_simple(nominator_name, columns, mode=nominator_search_mode))
            primary_nominator_ids = set(matching_nominators['nominator_id'].unique().to_list())
        
        if nominee_name and len(nominee_name) > 0:
            columns = ['nominee_name']
            matching_nominees = df_filtered.filter(search_in_columns_simple(nominee_name, columns, mode=nominee_search_mode))
            primary_nominee_ids = set(matching_nominees['nominee_id'].unique().to_list())
        
        # Combine all primary person IDs
        primary_person_ids = primary_nominator_ids | primary_nominee_ids
        
        if primary_person_ids:
            # Step 2: Find ALL edges connected to primary persons
            # (either as nominator OR as nominee)
            df_filtered = df_filtered.filter(
                pl.col('nominator_id').is_in(list(primary_person_ids)) |
                pl.col('nominee_id').is_in(list(primary_person_ids))
            )
            
            # Step 3: Mark which nodes are primary vs secondary
            df_filtered = df_filtered.with_columns([
                pl.col('nominator_id').is_in(list(primary_person_ids)).alias('is_primary_nominator'),
                pl.col('nominee_id').is_in(list(primary_person_ids)).alias('is_primary_nominee')
            ])
        else:
            # No matching persons found - add empty marker columns
            df_filtered = df_filtered.with_columns([
                pl.lit(False).alias('is_primary_nominator'),
                pl.lit(False).alias('is_primary_nominee')
            ])
    else:
        # No name filter or expansion disabled - apply traditional filtering
        if nominator_name and len(nominator_name) > 0:
            columns = ['nominator_name']
            df_filtered = df_filtered.filter(search_in_columns_simple(nominator_name, columns, mode=nominator_search_mode))
        
        if nominee_name and len(nominee_name) > 0:
            columns = ['nominee_name']
            df_filtered = df_filtered.filter(search_in_columns_simple(nominee_name, columns, mode=nominee_search_mode))
        
        # Add marker columns (all True since they passed the filter)
        df_filtered = df_filtered.with_columns([
            pl.lit(True).alias('is_primary_nominator'),
            pl.lit(True).alias('is_primary_nominee')
        ])

    return df_filtered

# ============================================================================
# TRANSFORMATION TO EDGES
# ============================================================================

def transform_to_edges(df_nominations):

    edges_list = []
    skipped_nominations = 0
    skipped_nominations_list = []

    for row in df_nominations.iter_rows(named=True):
        nomination_id = row['nomination_id']
        year = row['nomination_year']
        category = row['nomination_category_from_title']
        motivation = row['nomination_motivation']

        nominator_id = row['nominator_1_id']
        nominator_name = row['nominator_1_name']
        nominator_gender = row['nominator_1_gender']
        nominator_country = row['nominator_1_country']
        nominator_prizes = row['nominator_1_awarded_prizes']

        # Skip if nominator ID is invalid
        if not is_valid_value(nominator_id):
            skipped_nominations += 1
            skipped_nominations_list.append(row)
            continue

        # Process all possible nominees; first get colum names, then get contents of that column for the current row
        for nominee_num in range(1, max_nominees + 1):
            nominee_id_col = f'nominee_{nominee_num}_id'
            nominee_name_col = f'nominee_{nominee_num}_name'
            nominee_gender_col = f'nominee_{nominee_num}_gender'
            nominee_country_col = f'nominee_{nominee_num}_country'
            nominee_prizes_col = f'nominee_{nominee_num}_awarded_prizes'

            nominee_id = row.get(nominee_id_col)
            nominee_name = row.get(nominee_name_col)
            nominee_gender = row.get(nominee_gender_col)
            nominee_country = row.get(nominee_country_col)
            nominee_prizes = row.get(nominee_prizes_col)

            # Check if this nominee exists and has valid data
            if not is_valid_value(nominee_name) or not is_valid_value(nominee_id):
                continue

            # Create edge dictionary
            edge = {
                'nomination_id': nomination_id,
                'year': year,
                'category': category,
                'motivation': motivation,
                'nominator_id': int(nominator_id),
                'nominator_name': nominator_name,
                'nominator_gender': nominator_gender,
                'nominator_country': nominator_country if is_valid_value(nominator_country) else 'Unknown',
                'nominator_prizes': nominator_prizes,
                'nominee_id': int(nominee_id),
                'nominee_name': nominee_name,
                'nominee_gender': nominee_gender,
                'nominee_country': nominee_country if is_valid_value(nominee_country) else 'Unknown',
                'nominee_prizes': nominee_prizes
            }

            edges_list.append(edge)

    return edges_list, skipped_nominations, skipped_nominations_list


def clean_edges(edges_list):
    df_edges = pl.DataFrame(edges_list) # transform to Polars DataFrame

    df_match_edges_country = pl.read_csv('edges_country_match.csv', separator=';', encoding='utf8')
    df_coordinates = pl.read_csv('countries_with_coordinates.csv', separator=';', encoding='utf8')

    country_mapping = dict(zip(
        df_match_edges_country["CountryEdges"],
        df_match_edges_country["CountryRegular"]
    ))

    df_edges = df_edges.with_columns(
        pl.col("nominator_country").replace(country_mapping),
        pl.col("nominee_country").replace(country_mapping)
    )

    df_edges = df_edges.join(
        df_coordinates.select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominator_lat"),
            pl.col("Longitude").alias("nominator_lon")
        ]),
        left_on="nominator_country",
        right_on="Country",
        how="left"  
    )

    df_edges = df_edges.join(
        df_coordinates.select([
            pl.col("Country"),
            pl.col("Latitude").alias("nominee_lat"),
            pl.col("Longitude").alias("nominee_lon")
        ]),
        left_on="nominee_country",
        right_on="Country",
        how="left"
    )

    return df_edges

# ============================================================================
# STATISTICS
# ============================================================================

def statistics_edges(df_edges, skipped_nominations, skipped_nominations_list):

    print(f"\n{len(df_edges)} edges created from {len(df_nominations)} nominations")
    if skipped_nominations > 0:
        print(f"Skipped {skipped_nominations} nominations due to missing nominator ID")
    print(f"Unique nominators: {df_edges['nominator_id'].n_unique()}")
    print(f"Unique nominees: {df_edges['nominee_id'].n_unique()}")
    
    skipped_ids = [row['nomination_id'] for row in skipped_nominations_list]
    print("Skipped nomination IDs:", skipped_ids)
    
    # Group nominations statistics
    group_nominations = df_edges.group_by('nomination_id').agg(pl.len().alias('count'))
    max_nominees_single = group_nominations['count'].max()
    group_nominations_filtered = group_nominations.filter(pl.col('count') > 1)
    
    print(f"Largest number of nominees in a single nomination: {max_nominees_single}")
    print(f"Group nominations (1:n): {len(group_nominations_filtered)}")
    print(f"Single nominations (1:1): {len(group_nominations) - len(group_nominations_filtered)}")

    print("\nSample edges:")
    print(df_edges.select(['nomination_id', 'nominator_name', 'nominee_name', 'year', 'category']).head(10))



# ============================================================================
# BUILD NETWORKX GRAPH - OPTIMIZED VERSION
# ============================================================================

def build_network_graph(df_edges=df_edges, df_nominations=df_nominations):
    
    # Use MultiDiGraph to allow multiple edges between same nodes
    # (same person can nominate another person in different years)
    G = nx.MultiDiGraph()
    
    # ========================================================================
    # STEP 1: Pre-compute laureate IDs (MUCH faster than repeated checks)
    # ========================================================================
    
    #print("Pre-computing laureate IDs...")
    laureate_ids = set()
    
    # Cast all nominee_id columns to int once
    for i in range(1, max_nominees + 1):
        col = f'nominee_{i}_id'
        if col in df_nominations.columns:
            df_nominations = df_nominations.with_columns(
                pl.col(col).cast(pl.Int64, strict=False)
            )
    
    # Find all laureates by checking awarded_prizes columns
    for i in range(1, max_nominees + 1):
        id_col = f'nominee_{i}_id'
        award_col = f'nominee_{i}_awarded_prizes'
        
        if id_col in df_nominations.columns and award_col in df_nominations.columns:
            # Filter rows where this nominee won
            laureates = df_nominations.filter(
                pl.col(award_col).is_not_null() & (pl.col(award_col) != '')
            ).select(id_col).unique()
            
            # Add to set
            for laureate_id in laureates[id_col].to_list():
                if laureate_id is not None:
                    laureate_ids.add(laureate_id)
    
   
    # ========================================================================
    # STEP 2: Build person data from edges (single pass)
    # ========================================================================
    
    # print("Building person data...")
    all_people = {}
    
    # Track which persons are primary (matched the name filter directly)
    primary_person_ids = set()
    
    # Convert to pandas for faster iteration (Polars iter_rows is slow)
    df_edges_pd = df_edges.to_pandas()
    
    # Check if primary marker columns exist
    has_primary_markers = 'is_primary_nominator' in df_edges_pd.columns
    
    for _, row in df_edges_pd.iterrows():
        nominator_id = row['nominator_id']
        nominee_id = row['nominee_id']
        category = row['category']
        
        # Track primary persons
        if has_primary_markers:
            if row.get('is_primary_nominator', False):
                primary_person_ids.add(nominator_id)
            if row.get('is_primary_nominee', False):
                primary_person_ids.add(nominee_id)
        
        # Add/update nominator
        if nominator_id not in all_people:
            all_people[nominator_id] = {
                'name': row['nominator_name'],
                'country': row['nominator_country'],
                'type': 'nominator',
                'categories': {category},
                'is_laureate': nominator_id in laureate_ids
            }
        else:
            all_people[nominator_id]['categories'].add(category)
        
        # Add/update nominee
        if nominee_id not in all_people:
            all_people[nominee_id] = {
                'name': row['nominee_name'],
                'country': row['nominee_country'],
                'type': 'nominee',
                'categories': {category},
                'is_laureate': nominee_id in laureate_ids
            }
        else:
            all_people[nominee_id]['categories'].add(category)
            # Update type if person is both nominator and nominee
            if all_people[nominee_id]['type'] == 'nominator':
                all_people[nominee_id]['type'] = 'both'
    
    # ========================================================================
    # STEP 3: Determine main category and finalize person data
    # ========================================================================
    
    #print("Computing main categories...")
    
    # Count category appearances for each person
    category_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in df_edges_pd.iterrows():
        nominator_id = row['nominator_id']
        nominee_id = row['nominee_id']
        category = row['category']
        
        category_counts[nominator_id][category] += 1
        category_counts[nominee_id][category] += 1
    
    # Set main_category and is_primary for each person
    for person_id, person_data in all_people.items():
        # Convert set to list
        person_data['categories'] = list(person_data['categories'])
        
        # Set main_category to most frequent
        if person_id in category_counts:
            person_data['main_category'] = max(
                category_counts[person_id], 
                key=category_counts[person_id].get
            )
        else:
            person_data['main_category'] = 'Unknown'
        
        # Mark if this person is primary (matched the name filter)
        # If no primary markers exist, all persons are considered primary
        person_data['is_primary'] = person_id in primary_person_ids if primary_person_ids else True
    
    # ========================================================================
    # STEP 4: Add nodes to graph (batch operation)
    # ========================================================================
    
    #print("Adding nodes to graph...")
    node_data = [(person_id, person_attrs) for person_id, person_attrs in all_people.items()]
    G.add_nodes_from(node_data)
    
  
    # ========================================================================
    # STEP 5: Add edges (batch operation)
    # ========================================================================
    
    #print("Adding edges to graph...")
    edges_to_add = []
    
    for _, row in df_edges_pd.iterrows():
        edges_to_add.append((
            row['nominator_id'],
            row['nominee_id'],
            {
                'nomination_id': row['nomination_id'],
                'year': row['year'],
                'category': row['category'],
                'motivation': row['motivation']
            }
        ))
    
    G.add_edges_from(edges_to_add)
    
   
    # ========================================================================
    # STEP 6: Add co-nominee information
    # ========================================================================
    
    #print("Computing co-nominees...")
    
    # Group nominees by nomination_id
    nomination_groups = defaultdict(list)
    for _, row in df_edges_pd.iterrows():
        nomination_groups[row['nomination_id']].append(row['nominee_name'])
    
    # Add co-nominees info to edges
    for u, v, data in G.edges(data=True):
        nomination_id = data['nomination_id']
        data['co_nominees'] = nomination_groups[nomination_id]
        data['is_group_nomination'] = len(nomination_groups[nomination_id]) > 1
    
    # Statistics
    group_edges = sum(1 for u, v, d in G.edges(data=True) if d['is_group_nomination'])
    single_edges = G.number_of_edges() - group_edges
    

    #print("Graph building complete!")
    
    return G


# ============================================================================
# LAYOUT CALCULATION - OPTIMIZED VERSION
# ============================================================================

def calculate_layout(G, LAYOUT_ALGORITHM, sfdp_k_value=0.3, sfdp_rf_value=1.0, sfdp_overlap="scale"):
    
    n_nodes = G.number_of_nodes()
    #print(f"Calculating layout for {n_nodes} nodes using {LAYOUT_ALGORITHM}...")
    
    # ========================================================================
    # PRE-COMPUTE NODE CATEGORIES (once, not per algorithm)
    # ========================================================================
    
    laureates = [n for n, d in G.nodes(data=True) if d.get('is_laureate', False)]
    both = [n for n, d in G.nodes(data=True) if d['type'] == 'both' and not d.get('is_laureate', False)]
    nominees = [n for n, d in G.nodes(data=True) if d['type'] == 'nominee' and not d.get('is_laureate', False)]
    nominators = [n for n, d in G.nodes(data=True) if d['type'] == 'nominator']
    
  
    # ========================================================================
    # LAYOUT ALGORITHMS
    # ========================================================================
    
    if LAYOUT_ALGORITHM == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    
    elif LAYOUT_ALGORITHM == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    elif LAYOUT_ALGORITHM == 'spectral':
        pos = nx.spectral_layout(G)
    
    elif LAYOUT_ALGORITHM == 'circular':
        pos = nx.circular_layout(G)
    
    elif LAYOUT_ALGORITHM == 'shell':
        shells = [laureates, both]
        other_nodes = [n for n in G.nodes() if n not in laureates and n not in both]
        shells.append(other_nodes)
        shells = [s for s in shells if s]
        pos = nx.shell_layout(G, nlist=shells)
    
    elif LAYOUT_ALGORITHM == 'spring_communities':
        # Better for clustered networks
        pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'hierarchical':
        # Layer by node type
        pos = {}
        layers = [laureates, both, nominees, nominators]
        y_positions = [3, 2, 1, 0]
        
        for layer, y_pos in zip(layers, y_positions):
            if not layer:
                continue
            spacing = 2
            x_offset = -(len(layer) - 1) / 2  # Center the layer
            for i, node in enumerate(layer):
                pos[node] = ((i + x_offset) * spacing, y_pos)
    
    elif LAYOUT_ALGORITHM == 'community':
        import networkx.algorithms.community as nx_comm
        
        #print("Detecting communities...")
        communities = list(nx_comm.louvain_communities(G.to_undirected()))
        #print(f"Detected {len(communities)} communities")
        
        pos = {}
        for i, community in enumerate(communities):
            # Position communities in circle
            angle = 2 * np.pi * i / len(communities)
            center_x = 15 * np.cos(angle)
            center_y = 15 * np.sin(angle)
            
            # Layout within community (much faster on small subgraphs)
            community_list = list(community)
            if len(community_list) > 1:
                subgraph = G.subgraph(community_list)
                # Reduce iterations for large communities
                iters = min(50, max(20, 100 // len(communities)))
                sub_pos = nx.spring_layout(subgraph, scale=4, iterations=iters, seed=42)
            else:
                sub_pos = {community_list[0]: (0, 0)}
            
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + center_x, y + center_y)
    
    elif LAYOUT_ALGORITHM == 'fruchterman_reingold':
        # Similar to spring but with different physics
        pos = nx.fruchterman_reingold_layout(G, k=3, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'multipartite':
        # Separate columns by node type
        subset_key = 'layer'
        
        # Assign layers (reuse pre-computed lists)
        for node in laureates:
            G.nodes[node][subset_key] = 0
        for node in both:
            G.nodes[node][subset_key] = 1
        for node in nominees:
            G.nodes[node][subset_key] = 2
        for node in nominators:
            G.nodes[node][subset_key] = 3
        
        pos = nx.multipartite_layout(G, subset_key=subset_key, scale=5)
    
    elif LAYOUT_ALGORITHM == 'graphviz_neato':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_osage':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='osage')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_patchwork':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='patchwork')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_fdp':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='fdp')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    
    elif LAYOUT_ALGORITHM == 'graphviz_sfdp':
        try:
            args_str = f'-GK={sfdp_k_value} -Grepulsiveforce={sfdp_rf_value} -Goverlap={sfdp_overlap}'
            # print(f"[SFDP] Using parameters: {args_str}")
            pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp', args=args_str)
        except Exception as e:
            print(f"[SFDP] Error: {e}")
            return calculate_layout(G, 'by_category')
   

    elif LAYOUT_ALGORITHM == 'graphviz_circo':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='circo')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'graphviz_twopi':
        # Force-directed with good clustering
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'by_category':
        # Group by scientific category - FAST for large graphs
        #print("Grouping by category...")
        
        # Group nodes by category (reuse node data)
        categories = {}
        for node, data in G.nodes(data=True):
            cat = data.get('main_category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(node)
        
        # print(f"Found {len(categories)} categories")
        
        pos = {}
        num_cats = len(categories)
        
        for i, (cat, nodes) in enumerate(categories.items()):
            # Calculate center for this category
            angle = 2 * np.pi * i / num_cats
            center_x = 20 * np.cos(angle)
            center_y = 20 * np.sin(angle)
            
            # Layout within category (fast on subgraphs)
            if len(nodes) > 1:
                subgraph = G.subgraph(nodes)
                # Scale iterations with subgraph size
                iters = min(50, max(20, 1000 // len(nodes)))
                sub_pos = nx.spring_layout(subgraph, scale=5, k=2, iterations=iters, seed=42)
            else:
                sub_pos = {nodes[0]: (0, 0)}
            
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + center_x, y + center_y)
    
    elif LAYOUT_ALGORITHM == 'bipartite':
        # Only works if graph is actually bipartite
        nominator_nodes = set(nominators + both)
        nominee_nodes = set(nominees + both)
        
        # Set bipartite attribute
        for node in nominator_nodes:
            G.nodes[node]['bipartite'] = 0
        for node in nominee_nodes:
            G.nodes[node]['bipartite'] = 1
        
        pos = nx.bipartite_layout(G, nominator_nodes, scale=5)
    
    elif LAYOUT_ALGORITHM == 'random':
        pos = nx.random_layout(G, seed=42)
    
    else:
        print(f"Unknown layout algorithm: {LAYOUT_ALGORITHM}")
        print("Falling back to 'by_category' (good default for large graphs)")
        return calculate_layout(G, 'by_category')
    
    #print(f"Layout calculation complete!")
    return pos



# ============================================================================
# VISUALIZATION FUNCTION - OPTIMIZED VERSION
# ============================================================================

def create_network_figure(G, pos, df_edges, highlighted_person_id=None, debug=False):
    """
    Create Plotly network figure with optimized data preparation
    """
    
    if debug:
        print(f"\n[DEBUG] create_network_figure called (highlighted_person_id={highlighted_person_id})")
    
    import time
    start_time = time.time()
    
    # Remove nodes from pos that are not in G
    pos = {k: v for k, v in pos.items() if k in G.nodes}
    
    # ========================================================================
    # STEP 1: Pre-build lookup structures (CRITICAL for performance)
    # ========================================================================
    
    if debug:
        print("Building lookup structures...")
    
    # Build node connections lookup (instead of iterating edges for each node)
    node_out_edges = defaultdict(list)  # node -> list of (target, edge_data)
    node_in_edges = defaultdict(list)   # node -> list of (source, edge_data)
    
    # Build nomination groups lookup
    edges_by_nomination = defaultdict(list)
    nomination_data = {}  # Store first edge data per nomination
    
    for u, v, data in G.edges(data=True):
        nomination_id = data['nomination_id']
        
        # Cache connections
        node_out_edges[u].append((v, data))
        node_in_edges[v].append((u, data))
        
        # Cache nomination grouping
        edges_by_nomination[nomination_id].append((u, v, data))
        
        # Store first edge data for hover info
        if nomination_id not in nomination_data:
            nomination_data[nomination_id] = data
    
    if debug:
        print(f"  Built lookups in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 2: Determine highlighted nominations (if any)
    # ========================================================================
    
    highlighted_nominations = set()
    if highlighted_person_id is not None:
        # Use cached connections instead of iterating all edges
        for target, data in node_out_edges[highlighted_person_id]:
            highlighted_nominations.add(data['nomination_id'])
        for source, data in node_in_edges[highlighted_person_id]:
            highlighted_nominations.add(data['nomination_id'])
    
   # ========================================================================
    # STEP 3: Create edge traces (batch processing)
    # ========================================================================

    # ============================================================================
    # VISUAL CONFIGURATION - Adjust styling here
    # ============================================================================

    # Opacity configuration (applies to both nodes and edges)
    OPACITY_CONFIG = {
        'base': 0.8,              # No highlighting active
        'highlighted': 1.0,        # Highlighted nodes/edges
        'not_highlighted': 0.4    # Dimmed nodes/edges when highlighting active
    }

    # Highlighting colors
    HIGHLIGHT_CONFIG = {
        'color': cf.c_brand_color_acc,
        'non_highlighted_color': 'darkgrey'
    }

    # ============================================================================

    if debug:
        print("Creating edge traces...")

    # Category colors for edges
    category_colors_edges = {
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Physiology or Medicine': cf.c_medicine,
        'Medicine': cf.c_medicine,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace,
        'Economic Sciences': cf.c_economics,
        'Economics': cf.c_economics,
    }

    edge_traces = []

    for nomination_id, edges in edges_by_nomination.items():
        edge_x = []
        edge_y = []
        
        for u, v, data in edges:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        if not edge_x:
            continue
        
        # Use cached nomination data
        first_edge_data = nomination_data[nomination_id]
        co_nominees_str = ", ".join(first_edge_data['co_nominees'])
        
        # Handle None in motivation
        motivation = first_edge_data['motivation']
        motivation_text = (
            "No motivation provided" if motivation is None 
            else (str(motivation)[:100] + ("..." if len(str(motivation)) > 100 else ""))
        )
        
        # Determine styling
        is_highlighted = nomination_id in highlighted_nominations
        is_group = first_edge_data['is_group_nomination']
        edge_category = first_edge_data['category']
        
        # Get category color
        base_edge_color = category_colors_edges.get(edge_category, cf.c_grey)
        
        if is_highlighted:
            line_width = 5
            opacity = OPACITY_CONFIG['highlighted']
            line_color = HIGHLIGHT_CONFIG['color']
        elif highlighted_person_id is not None:
            line_width = 1
            opacity = OPACITY_CONFIG['not_highlighted']
            line_color = HIGHLIGHT_CONFIG['non_highlighted_color']
        else:
            line_width = 3 if is_group else 2
            opacity = OPACITY_CONFIG['base']
            line_color = base_edge_color
        
        hover_text = (
            f"<b>Nomination ID: {nomination_id}</b><br>"
            f"Year: {first_edge_data['year']}<br>"
            f"Category: {first_edge_data['category']}<br>"
            f"Nominees: {co_nominees_str}<br>"
            f"Motivation: {motivation_text}"
        )
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=line_width, color=line_color),
            hoverinfo='text',
            text=hover_text,
            opacity=opacity,
            showlegend=False,
            meta={
                'type': 'edge',
                'nomination_id': nomination_id,
                'category': edge_category,
                'connected_persons': list(set([u for u, v, d in edges] + [v for u, v, d in edges]))
            }
        )
        edge_traces.append(edge_trace)

    if debug:
        print(f"  Created {len(edge_traces)} edge traces in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 4: Pre-build node hover texts (HUGE optimization)
    # ========================================================================
    
    if debug:
        print("Building node hover texts...")
    
    # Category colors
    category_colors = {
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Physiology or Medicine': cf.c_medicine,
        'Medicine': cf.c_medicine,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace,
        'Economic Sciences': cf.c_economics,
        'Economics': cf.c_economics,
    }
    
    # Build hover texts for ALL nodes at once
    node_hover_cache = {}
    
    for node, node_data in G.nodes(data=True):
        if node not in pos or 'name' not in node_data:
            continue
        
        # Use cached connections (MUCH faster than iterating all edges)
        nominated_list = []
        for target, data in node_out_edges[node]:
            if target in G.nodes and 'name' in G.nodes[target]:
                nominated_list.append({
                    'name': G.nodes[target]['name'],
                    'year': data['year'],
                    'category': data['category']
                })
        
        was_nominated_list = []
        for source, data in node_in_edges[node]:
            if source in G.nodes and 'name' in G.nodes[source]:
                was_nominated_list.append({
                    'name': G.nodes[source]['name'],
                    'year': data['year'],
                    'category': data['category']
                })
        
        # Build hover text
        hover_lines = [
            f"<b>{node_data['name']}</b>",
            f"Country: {node_data['country']}",
            f"Category: {node_data.get('main_category', 'Unknown')}",
        ]
        
        # Add status indicator for primary vs secondary nodes
        is_primary = node_data.get('is_primary', True)
        if not is_primary:
            hover_lines.append("<i>(Connected via filter match)</i>")
        
        hover_lines.append("")  # Empty line
        
        if node_data.get('is_laureate', False):
            hover_lines.insert(3, "Status: Nobel Laureate ⭐")
        
        if nominated_list:
            hover_lines.append("<b>Nominated the following persons:</b>")
            for nom in nominated_list[:5]:
                hover_lines.append(f"  - {nom['name']}, {nom['year']}, {nom['category']}")
            if len(nominated_list) > 5:
                hover_lines.append(f"  ... and {len(nominated_list) - 5} more")
            hover_lines.append("")
        
        if was_nominated_list:
            hover_lines.append("<b>Was nominated by the following persons:</b>")
            for nom in was_nominated_list[:5]:
                hover_lines.append(f"  - {nom['name']}, {nom['year']}, {nom['category']}")
            if len(was_nominated_list) > 5:
                hover_lines.append(f"  ... and {len(was_nominated_list) - 5} more")
        
        hover_lines.append("")
        hover_lines.append("<i>Click to highlight</i>")
        
        node_hover_cache[node] = "<br>".join(hover_lines)
    
    if debug:
        print(f"  Built {len(node_hover_cache)} hover texts in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 5: Create node traces (using cached data)
    # ========================================================================

    # Node styling configuration by type
    NODE_STYLES = {
        'laureate': {
            'symbol': 'circle',
            'base_color': None,  # None = use category color
            'size': 12,
            'line_width': 1,
            'line_color': 'black'
        },
        'both': {
            'symbol': 'circle',
            'base_color': 'grey',
            'size': 10,
            'line_width': 1,
            'line_color': 'black'
        },
        'nominator': {
            'symbol': 'circle',
            'base_color': 'lightgrey',
            'size': 10,
            'line_width': 1,
            'line_color': 'grey'
        },
        'nominee': {
            'symbol': 'circle',
            'base_color': 'grey',
            'size': 10,
            'line_width': 1,
            'line_color': 'lightgrey'
        }
    }

    # Note: OPACITY_CONFIG and HIGHLIGHT_CONFIG are defined in STEP 3

    if debug:
        print("Creating node traces...")

    # Create nodes_by_symbol dict dynamically from NODE_STYLES
    # Now with separate tracking for primary and secondary nodes
    nodes_by_symbol = {}
    for node_type, style in NODE_STYLES.items():
        symbol = style['symbol']
        if symbol not in nodes_by_symbol:
            nodes_by_symbol[symbol] = {
                'x': [], 'y': [], 'text': [], 'color': [], 'size': [], 'ids': [], 'is_primary': []
            }

    for node, node_data in G.nodes(data=True):
        if node not in pos or node not in node_hover_cache:
            continue
        
        x, y = pos[node]
        
        # Check if this is a primary node (matched the name filter directly)
        is_primary = node_data.get('is_primary', True)
        
        # Determine node type
        if node_data.get('is_laureate', False):
            node_type = 'laureate'
        elif node_data['type'] == 'both':
            node_type = 'both'
        elif node_data['type'] == 'nominator':
            node_type = 'nominator'
        else:
            node_type = 'nominee'
        
        # Get style configuration
        style = NODE_STYLES[node_type]
        
        # Determine color
        main_category = node_data.get('main_category', 'Unknown')
        is_highlighted = (node == highlighted_person_id)
        
        if is_highlighted:
            color = HIGHLIGHT_CONFIG['color']
        elif highlighted_person_id is not None:
            color = HIGHLIGHT_CONFIG['non_highlighted_color']
        elif style['base_color']:
            color = style['base_color']
        else:
            color = category_colors.get(main_category, cf.c_grey)
        
        # Adjust size for secondary nodes (smaller)
        node_size = style['size'] if is_primary else style['size'] * 0.7
        
        # Add to corresponding symbol dict
        symbol = style['symbol']
        nodes_by_symbol[symbol]['x'].append(x)
        nodes_by_symbol[symbol]['y'].append(y)
        nodes_by_symbol[symbol]['text'].append(node_hover_cache[node])
        nodes_by_symbol[symbol]['color'].append(color)
        nodes_by_symbol[symbol]['size'].append(node_size)
        nodes_by_symbol[symbol]['ids'].append(node)
        nodes_by_symbol[symbol]['is_primary'].append(is_primary)

    # Create separate traces for each symbol
    node_traces = []
    for symbol, data in nodes_by_symbol.items():
        if len(data['x']) > 0:
            # Get line config for this symbol
            line_config = next((s for s in NODE_STYLES.values() if s['symbol'] == symbol), NODE_STYLES['both'])
            
            # Determine opacity
            if highlighted_person_id is None:
                opacity = OPACITY_CONFIG['base']
            else:
                opacity = OPACITY_CONFIG['highlighted']
            
            trace = go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                hoverinfo='text',
                text=data['text'],
                customdata=data['ids'],
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    symbol=symbol,
                    opacity=opacity,
                    line=dict(
                        width=line_config['line_width'], 
                        color=line_config['line_color'] if line_config['line_color'] else data['color']
                    )
                ),
                showlegend=False,
                meta={
                    'type': 'node',
                    'symbol': symbol
                }
            )
            node_traces.append(trace)

    if debug:
        print(f"  Created {len(node_traces)} node traces in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 6: Create figure
    # ========================================================================
    
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Add legend entries for node types (invisible dummy traces)
    legend_traces = []
    
    # Nominator (only nominator, not nominee)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='lightgrey', symbol='circle', 
                   line=dict(width=1, color='grey')),
        name='Nominator',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Nominee (only nominee, not nominator)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='grey', symbol='circle',
                   line=dict(width=1, color='lightgrey')),
        name='Nominee',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Both (nominee and nominator)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='grey', symbol='circle',
                   line=dict(width=1, color='black')),
        name='Nominee & Nominator',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Laureate (colored by category)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                   line=dict(width=1, color='black')),
        name='Laureate',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add category legend entries
    category_colors = {
        'Medicine': cf.c_medicine,
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Economic Sciences': cf.c_economics,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace
    }
    
    for category, color in category_colors.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color=color),
            name=category,
            showlegend=True,
            hoverinfo='skip'
        ))
    
    fig.add_traces(legend_traces)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Legend",
                        method="relayout",
                        args=[{"showlegend": True}],
                        args2=[{"showlegend": False}]
                    )
                ],
                x=0.01,
                y=0.01,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=cf.c_grey,
                borderwidth=1
            )
        ],
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=cf.c_plot_background,
        template='nbl_light',
        height=800,
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=cf.c_brand_color_main),
        hoverlabel=dict(bgcolor=cf.c_plot_background, font_size=12, font_family="IBM Plex Sans"),
    )


    
    if debug:
        print(f"\n[DEBUG] Total time: {time.time() - start_time:.2f}s")
        print(f"  Edge traces: {len(edge_traces)}")
        print(f"  Node traces: {len(node_traces)}")
    
    return fig


# Update figure with highlights
# ============================================================================
# HELPER FUNCTIONS FOR FAST HIGHLIGHTING
# ============================================================================

def extract_graph_connections(G):
    """
    Extract connection information from NetworkX graph for storage in dcc.Store.
    Now includes edge data (years, categories) for each connection.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dict with connection info that can be stored in dcc.Store
    """
    connections = {}
    
    for node in G.nodes():
        # Get successors (people this person nominated) with edge data
        # MultiDiGraph: G.edges[u, v] returns dict of {key: edge_data}
        successors_data = []
        for successor in G.successors(node):
            # Get ALL edges from node to successor (multiple nominations possible)
            edge_dict = G.get_edge_data(node, successor)
            if edge_dict:
                for edge_key, edge_data in edge_dict.items():
                    successors_data.append({
                        'id': str(successor),
                        'name': G.nodes[successor].get('name', 'Unknown'),
                        'year': edge_data.get('year', 'Unknown'),
                        'category': edge_data.get('category', 'Unknown')
                    })
        
        # Get predecessors (people who nominated this person) with edge data
        predecessors_data = []
        for predecessor in G.predecessors(node):
            # Get ALL edges from predecessor to node (multiple nominations possible)
            edge_dict = G.get_edge_data(predecessor, node)
            if edge_dict:
                for edge_key, edge_data in edge_dict.items():
                    predecessors_data.append({
                        'id': str(predecessor),
                        'name': G.nodes[predecessor].get('name', 'Unknown'),
                        'year': edge_data.get('year', 'Unknown'),
                        'category': edge_data.get('category', 'Unknown')
                    })
        
        connections[str(node)] = {
            'successors': successors_data,
            'predecessors': predecessors_data,
            'node_data': {
                'name': G.nodes[node].get('name', 'Unknown'),
                'country': G.nodes[node].get('country', 'Unknown'),
                'main_category': G.nodes[node].get('main_category', 'Unknown'),
                'is_laureate': G.nodes[node].get('is_laureate', False),
            }
        }
    
    return connections


def update_network_highlighting(existing_figure, highlighted_person_id=None, graph_connections=None):
    """
    Fast update: Only change colors/opacity based on highlighted person.
    Does NOT recalculate layout!
    
    Args:
        existing_figure: The current figure dict
        highlighted_person_id: Person ID to highlight, or None to reset
        graph_connections: Dict with graph connection info (from extract_graph_connections)
    
    Returns:
        Updated figure with new styling
    """
    import copy
    
    # Use same opacity config as in create_network_figure
    OPACITY_CONFIG = {
        'base': 0.6,
        'highlighted': 1.0,
        'not_highlighted': 0.1
    }
    
    # Deep copy to avoid modifying original
    fig = copy.deepcopy(existing_figure)
    
    if highlighted_person_id is None or graph_connections is None:
        # Reset: restore base opacity
        for trace in fig['data']:
            if trace.get('meta', {}).get('type') == 'edge':
                trace['opacity'] = OPACITY_CONFIG['base']
            elif trace.get('meta', {}).get('type') == 'node':
                trace['marker']['opacity'] = OPACITY_CONFIG['base']
        
        return fig
    
    # Convert highlighted_person_id to string for comparison
    highlighted_person_id_str = str(highlighted_person_id)
    
    # Find all connected persons using the connections dict
    connected_persons = set()
    connected_persons.add(highlighted_person_id_str)
    
    if highlighted_person_id_str in graph_connections:
        for successor in graph_connections[highlighted_person_id_str]['successors']:
            connected_persons.add(successor['id'])
        
        for predecessor in graph_connections[highlighted_person_id_str]['predecessors']:
            connected_persons.add(predecessor['id'])
    
    # Update node styling
    for trace in fig['data']:
        if trace.get('meta', {}).get('type') == 'node':
            customdata_list = trace.get('customdata', [])
            marker_opacity = trace['marker'].get('opacity', OPACITY_CONFIG['base'])
            
            if isinstance(marker_opacity, (list, tuple)):
                new_opacities = list(marker_opacity)
            else:
                new_opacities = [marker_opacity] * len(customdata_list)
            
            for i, person_id in enumerate(customdata_list):
                person_id_str = str(person_id)
                
                if person_id_str in connected_persons:
                    new_opacities[i] = OPACITY_CONFIG['highlighted']
                else:
                    new_opacities[i] = OPACITY_CONFIG['not_highlighted']
            
            trace['marker']['opacity'] = new_opacities
        
        elif trace.get('meta', {}).get('type') == 'edge':
            # Check if edge connects highlighted persons
            connected_persons_list = trace.get('meta', {}).get('connected_persons', [])
            
            # If all persons in this edge are in the connected set
            if all(str(p) in connected_persons for p in connected_persons_list):
                trace['opacity'] = OPACITY_CONFIG['highlighted']
            else:
                trace['opacity'] = OPACITY_CONFIG['not_highlighted']
    
    return fig

def get_person_info_text(graph_connections, person_id):
    """Generate info text for clicked person with full nomination details including years"""
    
    if graph_connections is None:
        return "No graph data available"
    
    # Convert person_id to string for comparison (since JSON serialization converts to strings)
    person_id_str = str(person_id)
    
    if person_id_str not in graph_connections:
        return f"Person {person_id} not found in graph"
    
    person_info = graph_connections[person_id_str]
    person_data = person_info['node_data']
    name = person_data.get('name', 'Unknown')
    country = person_data.get('country', 'Unknown')
    category = person_data.get('main_category', 'Unknown')
    is_laureate = person_data.get('is_laureate', False)
    
    # Build info text
    info_parts = [f"**{name}**"]
    info_parts.append(f"**Country:** {country}")
    info_parts.append(f"**Category:** {category}")
    
    if is_laureate:
        info_parts.append("**Status:** Nobel Laureate Ã¢Â­Â")
    
    info_parts.append("")  # Empty line
    
    # Get nominated persons with details (year and name) - ALLE anzeigen
    successors = person_info.get('successors', [])
    if successors:
        info_parts.append(f"**Nominated {len(successors)} person(s):**")
        for succ in successors:  # Kein Limit mehr
            succ_name = succ.get('name', 'Unknown')
            succ_year = succ.get('year', '?')
            info_parts.append(f"- {succ_year}: {succ_name}")
    
    # Get nominators with details (year and name) - ALLE anzeigen
    predecessors = person_info.get('predecessors', [])
    if predecessors:
        info_parts.append("")  # Empty line
        info_parts.append(f"**Was nominated by {len(predecessors)} person(s):**")
        for pred in predecessors:  # Kein Limit mehr
            pred_name = pred.get('name', 'Unknown')
            pred_year = pred.get('year', '?')
            info_parts.append(f"- {pred_year}: {pred_name}")
    
    return "\n\n".join(info_parts)

# ============================================================================
# FUll Function
# ============================================================================

# filter nominations

def generate_network(data=df_nominations,
                     algorithm="graphviz_sfdp", sfdp_k_value=1.0, sfdp_rf_value=1.0, sfdp_overlap="prism",
                     categories="all", timerange_nomination=[1901, lastyearincluded-49], timerange_field="nomination",
                     nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, 
                     nominee_name="", nominee_search_mode="all", nominee_gender="female", nominee_country="all", nominee_islaureate=False, highlighted_person_id=None):

    # Step 1: Create Polars DataFrame with all edges

    # print(f"\n[ALGORITHM] {algorithm}")

    func_start_time = time.time()
    edges_list, skipped_nominations, skipped_nominations_list = transform_to_edges(data)
    df_edges = clean_edges(edges_list)
    # print(f"[TIME] Create Edges: {time.time() - func_start_time:.2f}s")


    # Step 2: Filter
    func_start_time = time.time()
    df_edges = filter_edges(df_edges, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    # print(f"[TIME] Filter: {time.time() - func_start_time:.2f}s")
    # print(f"[SIZE] Rows: {len(df_edges)}")

    # Step 3: Build network graph
    func_start_time = time.time()
    G = build_network_graph(df_edges, data)
    # print(f"[TIME] Build network graph: {time.time() - func_start_time:.2f}s")
    # print(f"[SIZE] Edges: {G.number_of_edges()}")

    # Step 4: Calculate layout
    func_start_time = time.time()
    pos = calculate_layout(G, algorithm, sfdp_k_value, sfdp_rf_value, sfdp_overlap,)
    # print(f"[TIME] Calculate layout: {time.time() - func_start_time:.2f}s")

    # Step 5: Generate plot
    func_start_time = time.time()
    initial_fig = create_network_figure(G, pos, df_edges, highlighted_person_id, debug=False)
    # print(f"[TIME] Generate plot: {time.time() - func_start_time:.2f}s")

    return initial_fig, G


def graph_to_cytoscape_elements(G, max_nodes=800):
    """
    Convert a nominations NetworkX graph into dash-cytoscape `elements`.

    Unlike the Plotly path there is NO server-side layout: Cytoscape.js positions
    the nodes in the browser. Category colors + laureate status are attached as CSS
    *classes* (see theme.CATEGORY_CLASS) so a theme toggle only swaps the stylesheet.
    Node size scales with degree; the highest-degree people are kept when the graph
    exceeds `max_nodes` so the browser stays responsive.
    """
    import theme as th

    # Degree per node (total nominations touched); cap to the busiest people.
    degree = dict(G.degree())
    keep = set(sorted(degree, key=degree.get, reverse=True)[:max_nodes]) if len(degree) > max_nodes else set(degree)

    nodes = []
    for nid in keep:
        attrs = G.nodes[nid]
        main_cat = attrs.get("main_category", "Unknown")
        classes = [th.CATEGORY_CLASS.get(main_cat, "cat-other")]
        if attrs.get("is_laureate"):
            classes.append("laureate")
        deg = degree.get(nid, 1)
        nodes.append({
            "data": {
                "id": str(nid),
                "label": attrs.get("name", str(nid)),
                "country": attrs.get("country", "Unknown"),
                "category": main_cat,
                "role": attrs.get("type", "nominee"),
                "is_laureate": bool(attrs.get("is_laureate", False)),
                "degree": deg,
                # 12..48 px by degree (sqrt keeps hubs from exploding).
                "size": 12 + min(36, (deg ** 0.5) * 6),
            },
            "classes": " ".join(classes),
        })

    edges = []
    seen = set()
    for u, v, data in G.edges(data=True):
        if u not in keep or v not in keep:
            continue
        cat = data.get("category", "Unknown")
        # Collapse parallel edges (same pair + category) into one line.
        key = (u, v, cat)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "data": {
                "source": str(u),
                "target": str(v),
                "category": cat,
                "year": data.get("year"),
            },
            "classes": th.CATEGORY_CLASS.get(cat, "cat-other"),
        })

    return nodes + edges


def generate_network_elements(data=df_nominations,
                     algorithm="cola", sfdp_k_value=1.0, sfdp_rf_value=1.0, sfdp_overlap="prism",
                     categories="all", timerange_nomination=[1901, lastyearincluded-49], timerange_field="nomination",
                     nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False,
                     nominee_name="", nominee_search_mode="all", nominee_gender="female", nominee_country="all", nominee_islaureate=False, highlighted_person_id=None):
    """
    dash-cytoscape counterpart of generate_network(): same edge-building + filtering +
    graph pipeline, but returns (elements, G) instead of (plotly_figure, G). The layout
    is done in the browser by Cytoscape, so no calculate_layout / create_network_figure.
    Extra kwargs (algorithm, sfdp_*) are accepted for call-signature parity and ignored.
    """
    edges_list, skipped_nominations, skipped_nominations_list = transform_to_edges(data)
    df_edges_local = clean_edges(edges_list)
    df_edges_local = filter_edges(df_edges_local, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    G = build_network_graph(df_edges_local, data)
    elements = graph_to_cytoscape_elements(G)
    return elements, G




#def generate_map_nominations(data=df_nominations, categories="all", timerange_nomination=[1901, 1902], timerange_field="nomination", nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, nominee_name="", nominee_search_mode="all", nominee_gender="all", nominee_country="all", nominee_islaureate=False, algorithm="graphviz_sfdp", sfdp_k_value=1.0, sfdp_rf_value=1.0, sfdp_overlap="prism"):
def generate_map_nominations(data=df_nominations,
                     algorithm="graphviz_sfdp", sfdp_k_value=1.0, sfdp_rf_value=1.0, sfdp_overlap="prism",
                     categories="all", timerange_nomination=[1901, lastyearincluded-49], timerange_field="nomination",
                     nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, 
                     nominee_name="", nominee_search_mode="all", nominee_gender="female", nominee_country="all", nominee_islaureate=False, highlighted_person_id=None):

    import time
    func_start = time.time()

    # Use pre-computed df_edges instead of re-transforming
    filtered_data = filter_edges(df_edges, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    filtered_data = filtered_data.filter(pl.col("nominator_lat").is_not_null() & pl.col("nominator_lon").is_not_null())
    # print(f"[MAP TIME] Filter: {time.time() - func_start:.2f}s")



    step_start = time.time()
    filtered_data = filtered_data.filter(
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
    # print(f"[MAP TIME] Coordinate filter: {time.time() - step_start:.2f}s, rows: {len(filtered_data)}")

    def interpolate_points_vectorized(lat1, lon1, lat2, lon2, num_points=30):
            """
            Vectorized geodesic interpolation - much faster than row-by-row.
            Returns arrays with None separators for Plotly line breaks.
            Handles dateline crossing by skipping paths that cross it.
            
            Parameters: lat1, lon1 (start), lat2, lon2 (end)
            """
            if len(lat1) == 0:
                return [], []
                
            all_lats = []
            all_lons = []
            
            for i in range(len(lat1)):
                # Check if this path crosses the dateline (check LONGITUDE difference)
                # lon1 and lon2 are the longitude arrays
                lon_diff = lon2[i] - lon1[i]
                
                # If longitude difference is large, we're crossing the dateline
                if abs(lon_diff) > 180:
                    # Skip this connection to avoid artifacts
                    continue
                
                # Convert to radians
                lat1_r, lon1_r = np.radians(lat1[i]), np.radians(lon1[i])
                lat2_r, lon2_r = np.radians(lat2[i]), np.radians(lon2[i])
                
                # Great circle distance
                d = 2 * np.arcsin(np.sqrt(
                    np.sin((lat2_r - lat1_r) / 2) ** 2 + 
                    np.cos(lat1_r) * np.cos(lat2_r) * np.sin((lon2_r - lon1_r) / 2) ** 2
                ))
                
                if np.isclose(d, 0):
                    # Straight line for zero distance
                    lats = np.linspace(lat1[i], lat2[i], num_points)
                    lons = np.linspace(lon1[i], lon2[i], num_points)
                else:
                    # Great circle interpolation
                    t_values = np.linspace(0, 1, num_points)
                    A = np.sin((1 - t_values) * d) / np.sin(d)
                    B = np.sin(t_values * d) / np.sin(d)
                    
                    x = A * np.cos(lat1_r) * np.cos(lon1_r) + B * np.cos(lat2_r) * np.cos(lon2_r)
                    y = A * np.cos(lat1_r) * np.sin(lon1_r) + B * np.cos(lat2_r) * np.sin(lon2_r)
                    z = A * np.sin(lat1_r) + B * np.sin(lat2_r)
                    
                    lats = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
                    lons = np.degrees(np.arctan2(y, x))
                
                all_lats.extend(lats.tolist())
                all_lons.extend(lons.tolist())
                # Add None to break the line between paths
                all_lats.append(None)
                all_lons.append(None)
            
            return all_lats, all_lons

    # Category to color mapping
    category_colors = {
        "Medicine": cf.c_medicine,
        "Physics": cf.c_physics,
        "Chemistry": cf.c_chemistry,
        "Economic Sciences": cf.c_economics,
        "Literature": cf.c_literature,
        "Peace": cf.c_peace
    }

    # Initialize the figure
    fig = go.Figure()

    # Add markers for nominators (unique locations)
    fig.add_trace(go.Scattermap(
        lon=filtered_data["nominator_lon"],
        lat=filtered_data["nominator_lat"],
        text=filtered_data["nominator_name"],
        hoverinfo="text",
        mode="markers",
        marker=go.scattermap.Marker(
            size=8,
            color=cf.c_grey,
            opacity=0.5
        ),
        name="Nominator"
    ))

    # Add markers for nominees
    fig.add_trace(go.Scattermap(
        lon=filtered_data["nominee_lon"],
        lat=filtered_data["nominee_lat"],
        text=filtered_data["nominee_name"],
        hoverinfo="text",
        mode="markers",
        marker=go.scattermap.Marker(
            size=8,
            color=cf.c_grey,
            opacity=0.7
        ),
        name="Nominee"
    ))

    # Create one trace per category (6 traces max - good performance with category colors)
    step_start = time.time()
    if len(filtered_data) > 0:
        # Get unique categories in the data
        unique_categories = filtered_data["category"].unique().to_list()
        
        for category in unique_categories:
            # Filter data for this category
            cat_data = filtered_data.filter(pl.col("category") == category)
            
            if len(cat_data) == 0:
                continue
            
            # Extract coordinate arrays for this category
            nom_lats = cat_data["nominator_lat"].to_numpy()
            nom_lons = cat_data["nominator_lon"].to_numpy()
            nee_lats = cat_data["nominee_lat"].to_numpy()
            nee_lons = cat_data["nominee_lon"].to_numpy()
            
            # Vectorized interpolation
            all_lats, all_lons = interpolate_points_vectorized(
                nom_lats, nom_lons, nee_lats, nee_lons, 
                num_points=30
            )
            
            # Get color for this category
            color = category_colors.get(category, cf.c_teal)
            
            # Add trace for this category
            fig.add_trace(go.Scattermap(
                lon=all_lons,
                lat=all_lats,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.6,
                hoverinfo="skip",
                name=category
            ))
    
    # print(f"[MAP TIME] Path generation: {time.time() - step_start:.2f}s")

    # Update layout for the mapbox visualization

    fig.update_layout(
        template='nbl_light',
        plot_bgcolor=cf.c_plot_background,
        title_text="Countries of Nominator & Nominee",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=30, lon=80),
            zoom=1.0,
        ),
        hoverlabel=dict(
                bgcolor=cf.c_hoverlabel_bg,
                font_size=12,
                font_family="IBM Plex Sans"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # print(f"[MAP TIME] Total: {time.time() - func_start:.2f}s")

    G = None  # No graph generated in this function, but needed for return consistency

    return fig, G


# Section?
##################################################################################################

# Overview Stats
#=================================================================================================


def generate_overview_stats(df_filtered_laureates=df_laureates, df_filtered_prizes=df_prizes):

    if df_filtered_laureates.shape[0] == 0 or df_filtered_prizes.shape[0] == 0:
        # print("generate_overview_stats: DataFrames are empty.")
        number_of_laureates = 0
        # print("Number of Laureates:", number_of_laureates)

        number_of_prizes = 0
        laureate_oldest_name = "None"
        laureate_oldest_age = "---"
        laureate_youngest_name = "None"
        laureate_youngest_age = "---"

        return number_of_laureates, number_of_prizes, laureate_oldest_name, laureate_oldest_age, laureate_youngest_name, laureate_youngest_age


    else:
        # Number of Prizes
        number_of_prizes = df_filtered_prizes.shape[0]

        # Number of Laureates#
        number_of_laureates = df_filtered_laureates.shape[0]

        # Calculate Youngest and Oldest laureate
        if df_filtered_prizes.shape[0] == 0:
            laureate_oldest_name = "None"
            laureate_oldest_age = ""
            laureate_youngest_name = "None"
            laureate_youngest_age = ""
        else:
            df_oldestyoungest_laureate = df_filtered_prizes[["AwardeeDisplayName", "OrganisationName", "BirthDate", "Prize0_AwardYear"]].copy()
            df_oldestyoungest_laureate = df_oldestyoungest_laureate[pd.isna(df_oldestyoungest_laureate["OrganisationName"])]
            df_oldestyoungest_laureate["BirthDate"] = df_oldestyoungest_laureate["BirthDate"].str.replace(r"-00-00", "-01-01", regex=True)
            df_oldestyoungest_laureate["BirthDate"] = pd.to_datetime(df_oldestyoungest_laureate["BirthDate"])

            # Convert Prize0_AwardYear to YYYY-12-10 format
            df_oldestyoungest_laureate["AwardDate"] = pd.to_datetime(
                df_oldestyoungest_laureate["Prize0_AwardYear"].astype(str) + "-12-10"
            )

            # Calculate the difference
            df_oldestyoungest_laureate["AgeAtAward"] = (
                df_oldestyoungest_laureate["AwardDate"] - df_oldestyoungest_laureate["BirthDate"]
            )

            from dateutil.relativedelta import relativedelta

            # Function to calculate exact age in years
            def calculate_exact_years(row):
                if pd.isna(row["BirthDate"]) or pd.isna(row["AwardDate"]):
                    return None  # Handle missing dates gracefully
                return relativedelta(row["AwardDate"], row["BirthDate"]).years

            # Apply the function to calculate age in years
            df_oldestyoungest_laureate["AgeAtAwardYears"] = df_oldestyoungest_laureate.apply(calculate_exact_years, axis=1)

            df_sorted = df_oldestyoungest_laureate.sort_values(by="AgeAtAward", ascending=False)

            laureate_oldest_name = df_sorted.iloc[0]["AwardeeDisplayName"]
            laureate_oldest_age = df_sorted.iloc[0]["AgeAtAwardYears"]

            #print(f"{laureate_oldest_name}: {laureate_oldest_age}")

            df_sorted = df_oldestyoungest_laureate.sort_values(by="AgeAtAward", ascending=True)

            laureate_youngest_name = df_sorted.iloc[0]["AwardeeDisplayName"]
            laureate_youngest_age = df_sorted.iloc[0]["AgeAtAwardYears"]

            #print(f"{laureate_youngest_name}: {laureate_youngest_age}")
        
            return number_of_laureates, number_of_prizes, laureate_oldest_name, laureate_oldest_age, laureate_youngest_name, laureate_youngest_age


##################################################################################################
# Save CSV Files
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