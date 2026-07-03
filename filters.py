"""
Shared filter helpers (standard_filter, extended_filter, chip-state translators).

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from data import (
    df_iso,
    df_laureates,
    df_prizes,
    lastyearincluded,
    pldf_laureates_enriched_redux_clean,
)


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


df_max_prize_count=count_per_country()


max_prize_count = df_max_prize_count['Count'].max()


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
