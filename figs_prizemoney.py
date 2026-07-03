"""
Prize money figures: amounts over time, total variance.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
from fractions import Fraction
from data import df_prizes, lastyearincluded
from filters import standard_filter


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
            # bgcolor/bordercolor come from the theme template (transparent, theme-aware)
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
