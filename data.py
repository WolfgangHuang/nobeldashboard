"""
Data loading, cleaning and enrichment. Runs once at import; every other
module imports its frames (and `lastyearincluded`) from here.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
import plotly.io as pio
from pathlib import Path


# All data files live next to the code; resolve them independently of the CWD
# (the old os.chdir() side effect is gone).
DATA_DIR = Path(__file__).resolve().parent


def data_path(name):
    """Absolute path of a data file in the project directory."""
    return str(DATA_DIR / name)


pio.templates['nbl_light'] = th.build_plotly_template(dark=False)


pio.templates['nbl_dark'] = th.build_plotly_template(dark=True)


pio.templates.default = 'nbl_light'


df_laureates_import = pd.read_csv(data_path('df_laureates.csv'), sep=';', index_col=0)


df_laureates_corrections = pd.read_csv(data_path('df_laureates_corrections.csv'), sep=';', index_col=0)


_lf_nominations = pl.scan_csv(data_path('nominations_full.csv'), separator=';', encoding='utf8')


_lf_edges = pl.scan_csv(data_path('df_edges.csv'), separator=';', encoding='utf8')


df_coordinates = pl.read_csv(data_path('countries_with_coordinates.csv'), separator=';', encoding='utf8')


df_match_edges_country = pl.read_csv(data_path('edges_country_match.csv'), separator=';', encoding='utf8')


_country_mapping = dict(zip(
    df_match_edges_country["CountryEdges"],
    df_match_edges_country["CountryRegular"]
))


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


df_nominations = _lf_nominations.collect()


_nominator_prizes = df_nominations.select([
    pl.col("nominator_1_id").cast(pl.Int64).alias("nominator_id"),
    pl.col("nominator_1_awarded_prizes").alias("nominator_prizes")
]).unique(subset=["nominator_id"]).drop_nulls(subset=["nominator_id"])


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


df_timegap = pd.read_csv(data_path('df_prize-publication-timegap.csv'), sep=';', encoding='UTF-8', index_col=0)


df_lifeexpectancy = pd.read_excel(data_path('df_life-expectancy.xlsx'))


df_pop = pd.read_excel(data_path("df_population.xlsx"))


df_movement_dwp = pd.read_csv(data_path('df_degree_institutions_work.csv'), sep=';', encoding='UTF-8', index_col=0)


df_movement_dwp = df_movement_dwp.fillna('None')


df_ethnicity = pd.read_csv(data_path('df_ethnicity.csv'), sep=';', index_col=0)


df_religion = pd.read_csv(data_path('df_religion.csv'), sep=';', index_col=0)


df_iso = pd.read_csv(data_path('countries_iso2_iso3.csv'), sep=';')


df_fields = pd.read_csv(data_path("df_fields.csv"), sep=";")


df_laureates_import.update(df_laureates_corrections)


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


columns_to_modify = ['BirthCountryNow', 
                     'DeathCountryNow', 
                     'Prize0_Affiliation0_CountryNow', 
                     'Prize0_Affiliation1_CountryNow', 
                     'Prize0_Residence0_CountryNow', 
                     'Prize0_Residence1_CountryNow', 
                     'Prize1_Affiliation0_CountryNow', 
                     'Prize0_Affiliation2_CountryNow', 
                     'Prize0_Affiliation3_CountryNow']


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


df_laureates = replace_values_in_columns(df_laureates_import, columns_to_modify, replacement_dict)


df_laureates['Prize0_Category'] = df_laureates['Prize0_Category'].replace('Physiology or Medicine', 'Medicine')


df_laureates['Prize1_Category'] = df_laureates['Prize1_Category'].replace('Physiology or Medicine', 'Medicine')


df_laureates['Prize2_Category'] = df_laureates['Prize2_Category'].replace('Physiology or Medicine', 'Medicine')


df_nominations = df_nominations.with_columns(
    pl.col('nomination_category_from_title').str.replace('Physiology or Medicine', 'Medicine')
)


df_edges = df_edges.with_columns(
    pl.col('category').str.replace('Physiology or Medicine', 'Medicine')
)


df_laureates = pd.merge(df_laureates, df_religion[["Religion", "ReligionSubgroup"]], left_index=True, right_index=True, how="left")


df_laureates = pd.merge(df_laureates, df_ethnicity[["Ethnicity"]], left_index=True, right_index=True, how="left")


df_laureates = pd.merge(df_laureates, df_timegap[["PublicationYear", "PublicationSource", "PublicationTimegap"]], left_index=True, right_index=True, how="left")


df_laureates = pd.merge(df_laureates, df_movement_dwp[["ParCatDegreeInstitution", "ParCatDegreeCity", "ParCatDegreeCountry", "ParCatDegreeCityCountry", "ParCatWorkInstitution", "ParCatWorkCity", "ParCatWorkCountry", "ParCatWorkCityCountry", "ParCatPrizeInstitution", "ParCatPrizeCity", "ParCatPrizeCountry", "ParCatPrizeCityCountry"]], left_index=True, right_index=True, how="left")


df_prize1 = df_laureates.dropna(subset=['Prize1_AwardYear']).copy(deep=True)


df_prize2 = df_laureates.dropna(subset=['Prize2_AwardYear']).copy(deep=True)


prize0_columns = [col for col in df_laureates.columns if col.startswith('Prize0_')]


prize1_columns = [col for col in df_laureates.columns if col.startswith('Prize1_')]


prize2_columns = [col for col in df_laureates.columns if col.startswith('Prize2_')]


df_prize1.drop(columns=prize0_columns, inplace=True)  # Drop existing Prize0 columns before renaming Prize1_ to Prize0_


df_prize1.drop(columns=prize2_columns, inplace=True)  # Drop existing Prize2 columns before renaming Prize1_ to Prize0_


renamed_columns = [col.replace('Prize1_', 'Prize0_') for col in df_prize1.columns if col.startswith('Prize1_')]


df_prize1.rename(columns={old: new for old, new in zip(df_prize1.columns[df_prize1.columns.str.startswith('Prize1_')], renamed_columns)}, inplace=True)


df_prize2.drop(columns=prize0_columns, inplace=True)  # Drop existing Prize0 columns before renaming Prize2_ to Prize0_


df_prize2.drop(columns=prize1_columns, inplace=True)  # Drop existing Prize1 columns before renaming Prize2_ to Prize0_


renamed_columns = [col.replace('Prize2_', 'Prize0_') for col in df_prize2.columns if col.startswith('Prize2_')]


df_prize2.rename(columns={old: new for old, new in zip(df_prize2.columns[df_prize2.columns.str.startswith('Prize2_')], renamed_columns)}, inplace=True)


df_prizes = pd.concat([df_laureates, df_prize1, df_prize2], ignore_index=True)


df_prizes.drop(columns=prize1_columns, inplace=True)  # Drop Prize1 columns, which are now obsolete


df_prizes.drop(columns=prize2_columns, inplace=True)  # Drop Prize2 columns, which are now obsolete


df_prizes['Prize0_AwardYear'] = df_prizes['Prize0_AwardYear'].astype('Int64') # Convert from float to nullable int


df_laureates_enriched_full_clean = df_laureates.copy()


df_prizes_enriched_full_clean = df_laureates.copy()


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


pldf_laureates_enriched_redux_clean = pl.DataFrame(df_laureates_enriched_redux_clean)


pldf_prizes_enriched_redux_clean = pl.DataFrame(df_prizes_enriched_redux_clean)


lf_laureates = pldf_laureates_enriched_redux_clean.lazy()


lf_prizes = pldf_prizes_enriched_redux_clean.lazy()


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
