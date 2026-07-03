import requests
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_laureates_data():
    """
    Fetch all laureates data from Nobel Prize API, handling pagination.

    Returns:
        pd.DataFrame: DataFrame containing all laureates data with extracted fields
    """
    logger.info("Starting laureates data fetch...")

    # Step 1: Define the base URL for the API
    base_url = 'https://api.nobelprize.org/2.1/laureates'

    # Step 2: Make an initial GET request to retrieve the meta information
    try:
        initial_response = requests.get(base_url)
        initial_response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve initial metadata: {e}")
        return None

    initial_data = initial_response.json()
    meta = initial_data.get('meta', {})
    limit = meta.get('limit', 25)  # Extract the 'limit' from the meta; default to 25 if not present
    offset = meta.get('offset', 0)  # Extract the 'offset' from the meta; default to 0 if not present

    params = {
        'limit': limit,  # Now using the limit from the API's meta
        'offset': 0  # Start from the first laureate
    }

    all_data = []  # To store all the laureates' data

    # Step 3: Loop through the pages of the API
    while True:
        try:
            # Make the GET request to the API with the correct limit
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            laureates = data['laureates']

            # Break the loop if there are no more laureates to fetch
            if not laureates:
                break

            logger.info(f"Processing {len(laureates)} laureates at offset {params['offset']}")

            # Step 4: Extract specific fields from each laureate
            for laureate in laureates:
                laureate_data = {

                    # PERSONS
                    'ID_Laureate': laureate.get('id'),
                    'LaureateNameKnown': laureate.get('knownName', {}).get('en'),
                    'LaureateNameFirst': laureate.get('givenName', {}).get('en'),
                    'LaureateNameLast': laureate.get('familyName', {}).get('en'),
                    'LaureateNameFull': laureate.get('fullName', {}).get('en'),
                    'LaureateNamePenOriginal': laureate.get('penName'),
                    'Filename': laureate.get('fileName', {}),
                    'LaureateGender': laureate.get('gender'),

                    # ORGANIZATIONS
                    'OrganisationName': laureate.get('orgName', {}).get('en') if laureate.get('orgName') else None,
                    'OrganisationNameNative': laureate.get('nativeName', '') if laureate.get('nativeName') else None,
                    'OrganisationAcronym': laureate.get('acronym', '') if laureate.get('acronym') else None,
                    'OrganisationFoundedDate': laureate.get('founded', {}).get('date') if laureate.get('founded') else None,
                    'OrganisationFoundedCity': laureate.get('founded', {}).get('place', {}).get('city', {}).get('en') if laureate.get('founded') else None,
                    'OrganisationFoundedCityNow': laureate.get('founded', {}).get('place', {}).get('cityNow', {}).get('en') if laureate.get('founded') else None,
                    'OrganisationFoundedCityNowWiki': laureate.get('founded', {}).get('place', {}).get('cityNow', {}).get('sameAs', [])[0] if laureate.get('founded', {}).get('place', {}).get('cityNow', {}).get('sameAs') else None,
                    'OrganisationFoundedCountry': laureate.get('founded', {}).get('place', {}).get('country', {}).get('en') if laureate.get('founded') else None,
                    'OrganisationFoundedCountryNow': laureate.get('founded', {}).get('place', {}).get('countryNow', {}).get('en') if laureate.get('founded') else None,
                    'OrganisationFoundedNowWiki': laureate.get('founded', {}).get('place', {}).get('countryNow', {}).get('sameAs', [])[0] if laureate.get('founded', {}).get('place', {}).get('countryNow', {}).get('sameAs') else None,
                    'OrganisationFoundedContinent': laureate.get('founded', {}).get('place', {}).get('continent', {}).get('en') if laureate.get('founded') else None,

                    # BIRTH
                    'BirthDate': laureate.get('birth', {}).get('date'),
                    'BirthCity': laureate.get('birth', {}).get('place', {}).get('city', {}).get('en'),
                    'BirthCityNow': laureate.get('birth', {}).get('place', {}).get('cityNow', {}).get('en'),
                    'BirthCityNowWiki': laureate.get('birth', {}).get('place', {}).get('cityNow', {}).get('sameAs', [])[0] if laureate.get('birth', {}).get('place', {}).get('cityNow', {}).get('sameAs') else None,
                    'BirthCityNowLat': laureate.get('birth', {}).get('place', {}).get('cityNow', {}).get('latitude'),
                    'BirthCityNowLon': laureate.get('birth', {}).get('place', {}).get('cityNow', {}).get('longitude'),
                    'BirthCountry': laureate.get('birth', {}).get('place', {}).get('country', {}).get('en'),
                    'BirthCountryNow': laureate.get('birth', {}).get('place', {}).get('countryNow', {}).get('en'),
                    'BirthCountryNowWiki': laureate.get('birth', {}).get('place', {}).get('countryNow', {}).get('sameAs', [])[0] if laureate.get('birth', {}).get('place', {}).get('countryNow', {}).get('sameAs') else None,
                    'BirthCountryLat': laureate.get('birth', {}).get('place', {}).get('countryNow', {}).get('latitude'),
                    'BirthCountryLon': laureate.get('birth', {}).get('place', {}).get('countryNow', {}).get('longitude'),
                    'BirthContinent': laureate.get('birth', {}).get('place', {}).get('continent', {}).get('en'),

                    # DEATH
                    'DeathDate': laureate.get('death', {}).get('date'),
                    'DeathCity': laureate.get('death', {}).get('place', {}).get('city', {}).get('en'),
                    'DeathCityNow': laureate.get('death', {}).get('place', {}).get('cityNow', {}).get('en'),
                    'DeathCityNowWiki': laureate.get('death', {}).get('place', {}).get('cityNow', {}).get('sameAs', [])[0] if laureate.get('death', {}).get('place', {}).get('cityNow', {}).get('sameAs') else None,
                    'DeathCityLat': laureate.get('death', {}).get('place', {}).get('cityNow', {}).get('latitude'),
                    'DeathCityLon': laureate.get('death', {}).get('place', {}).get('cityNow', {}).get('longitude'),
                    'DeathCountry': laureate.get('death', {}).get('place', {}).get('country', {}).get('en'),
                    'DeathCountryNow': laureate.get('death', {}).get('place', {}).get('countryNow', {}).get('en'),
                    'DeathCountryNowWiki': laureate.get('death', {}).get('place', {}).get('countryNow', {}).get('sameAs', [])[0] if laureate.get('death', {}).get('place', {}).get('countryNow', {}).get('sameAs') else None,
                    'DeathCountryLat': laureate.get('death', {}).get('place', {}).get('countryNow', {}).get('latitude'),
                    'DeathCountryLon': laureate.get('death', {}).get('place', {}).get('countryNow', {}).get('longitude'),
                    'DeathContinent': laureate.get('death', {}).get('place', {}).get('continent', {}).get('en'),

                    # WIKI
                    'WikipediaSlug': laureate.get('wikipedia', {}).get('slug') if laureate.get('wikipedia') else None,
                    'WikipediaURL': laureate.get('wikipedia', {}).get('english') if laureate.get('wikipedia') else None,
                    'WikidataID': laureate.get('wikidata', {}).get('id') if laureate.get('wikidata') else None,
                    'WikidataURL': laureate.get('wikidata', {}).get('url') if laureate.get('wikidata') else None,

                }

                # PRIZES

                # Step 5: Loop through all Nobel prizes and add them to the laureate's data
                if 'nobelPrizes' in laureate:
                    for i, prize in enumerate(laureate['nobelPrizes']):
                        # Add each prize's award year and category as separate columns
                        laureate_data[f'Prize{i}_AwardYear'] = prize.get('awardYear')
                        laureate_data[f'Prize{i}_Category'] = prize.get('category', {}).get('en')
                        laureate_data[f'Prize{i}_SortOrder'] = prize.get('sortOrder')
                        laureate_data[f'Prize{i}_Portion'] = prize.get('portion')
                        laureate_data[f'Prize{i}_DateAwarded'] = prize.get('dateAwarded')
                        laureate_data[f'Prize{i}_Status'] = prize.get('prizeStatus')
                        laureate_data[f'Prize{i}_Amount'] = prize.get('prizeAmount')
                        laureate_data[f'Prize{i}_AmountAdjusted_'] = prize.get('prizeAmountAdjusted')
                        laureate_data[f'Prize{i}_Motivation'] = prize.get('motivation', {}).get('en')
                        laureate_data[f'Prize{i}_MotivationTop'] = prize.get('topMotivation', {}).get('en')

                        # Step 6: Loop through all affiliations linked to the current prize
                        if 'affiliations' in prize:
                            for j, affiliation in enumerate(prize['affiliations']):
                                # Extract affiliation data
                                laureate_data[f'Prize{i}_Affiliation{j}_Name'] = affiliation.get('name', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_NameNow'] = affiliation.get('nameNow', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_NameNative'] = affiliation.get('nativeName')
                                laureate_data[f'Prize{i}_Affiliation{j}_City'] = affiliation.get('city', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_CityNow'] = affiliation.get('cityNow', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_CityLatitude'] = affiliation.get('cityNow', {}).get('latitude')
                                laureate_data[f'Prize{i}_Affiliation{j}_CityLongitude'] = affiliation.get('cityNow', {}).get('longitude')
                                laureate_data[f'Prize{i}_Affiliation{j}_Country'] = affiliation.get('country', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_CountryNow'] = affiliation.get('countryNow', {}).get('en')
                                laureate_data[f'Prize{i}_Affiliation{j}_CountryLat'] = affiliation.get('countryNow', {}).get('latitude')
                                laureate_data[f'Prize{i}_Affiliation{j}_CountryLon'] = affiliation.get('countryNow', {}).get('longitude')
                                laureate_data[f'Prize{i}_Affiliation{j}_Continent'] = affiliation.get('continent', {}).get('en')

                        # Step 7: Loop through all residences linked to the current prize
                        if 'residences' in prize:
                            for k, residence in enumerate(prize['residences']):
                                # Extract residence data
                                laureate_data[f'Prize{i}_Residence{k}_City'] = residence.get('city', {}).get('en')
                                laureate_data[f'Prize{i}_Residence{k}_CityNow'] = residence.get('cityNow', {}).get('en')
                                laureate_data[f'Prize{i}_Residence{k}_Country'] = residence.get('country', {}).get('en')
                                laureate_data[f'Prize{i}_Residence{k}_CountryNow'] = residence.get('countryNow', {}).get('en')
                                laureate_data[f'Prize{i}_Residence{k}_Continent'] = residence.get('continent', {}).get('en')

                # Append the laureate data to the list
                all_data.append(laureate_data)

            # Step 6: Update the offset to fetch the next page of laureates
            params['offset'] += params['limit']

        except requests.RequestException as e:
            logger.error(f"Failed to retrieve data at offset {params['offset']}: {e}")
            break

    if not all_data:
        logger.error("No data retrieved")
        return None

    # Step 7: Convert the list of dictionaries to a DataFrame
    df_laureates = pd.DataFrame(all_data)

    # Create AwardeeDisplayName column
    df_laureates.insert(1, 'AwardeeDisplayName',
                       df_laureates['LaureateNameKnown'].replace('', pd.NA).fillna(df_laureates['OrganisationName']))

    # Sort by ID
    df_laureates['ID_Laureate'] = pd.to_numeric(df_laureates['ID_Laureate'], errors='coerce')
    df_laureates.sort_values(by='ID_Laureate', ascending=True, inplace=True)

    logger.info(f"Successfully retrieved {len(df_laureates)} laureates")
    return df_laureates

def save_laureates_data(df_laureates, output_dir="."):
    """
    Save laureates data to CSV and Excel files.

    Args:
        df_laureates (pd.DataFrame): DataFrame containing laureates data
        output_dir (str): Directory to save the files
    """
    if df_laureates is None or df_laureates.empty:
        logger.error("No data to save")
        return False

    try:
        cwd = os.getcwd()
        csv_path = os.path.join(cwd, "df_laureates.csv")
        excel_path = os.path.join(cwd, "df_laureates.xlsx")
        df_laureates.to_csv(csv_path, encoding="UTF-8", sep=";", index=False)
        df_laureates.to_excel(excel_path, index=False)

        logger.info(f"Data saved in working directory.")
        return True

    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        return False

if __name__ == "__main__":
    # For testing purposes
    df = get_all_laureates_data()
    if df is not None:
        print(f"Retrieved {len(df)} laureates with {len(df.columns)} columns")
        save_laureates_data(df)