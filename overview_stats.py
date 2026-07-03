"""
Headline KPI numbers for the Overview tab.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from data import df_laureates, df_prizes


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
