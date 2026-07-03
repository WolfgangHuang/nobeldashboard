"""
App settings and color constants for the Nobel Prize Dashboard.
Plot configurations live in plot_config.py, filter UI in filters_ui.py.
"""

##################################################################################################
# App Settings
##################################################################################################

# App-level configuration
APP_CONFIG = {
    'title': 'Nobel Laureate Data Dashboard',
    # assets/*.css is auto-served by Dash; no explicit stylesheet entries needed.
    'external_stylesheets': [],
    'suppress_callback_exceptions': True,
    'port': 8050,
    # Safe defaults for `python app.py`: opt in to the Werkzeug debugger / external
    # binding via the DEBUG / HOST env vars. Production runs via Gunicorn (app:server).
    'debug': False,
    'host': '127.0.0.1'
}

##################################################################################################
# Currency Conversion Rates (static values to avoid API dependencies)
##################################################################################################

# SEK to EUR conversion rate (approximate as of January 2025)
# 1 SEK ≈ 0.086 EUR (i.e., ~11.6 SEK per EUR)
SEK_TO_EUR_RATE = 0.086

# SEK to USD conversion rate (approximate as of January 2025)
# 1 SEK ≈ 0.092 USD (i.e., ~10.9 SEK per USD)
SEK_TO_USD_RATE = 0.092

# # Navigation configuration
# NAVIGATION_ITEMS = [
#     {"label": "Overview", "value": "overview", "url": "/overview", "icon": "material-symbols:overview-key"},
#     {"label": "Time", "value": "time", "url": "/time", "icon": "material-symbols:hourglass-empty"},
#     {"label": "Demography", "value": "demography", "url": "/demography", "icon": "material-symbols:public"},
#     {"label": "List Generator", "value": "listgenerator", "url": "/listgenerator", "icon": "material-symbols:list"},
#     {"label": "Data & References", "value": "data", "url": "/data", "icon": "material-symbols:data-table-outline-rounded"},
# ]





##################################################################################################
# Colors
##################################################################################################



### Colors HEX ###

c_black= '#27262c'
c_blue= '#26384b'
c_teal= '#476e71'
c_green= '#2f754e'
c_yellow= '#edae49'
c_orange= '#f28118'
c_red= '#b50603'
c_purple= '#9f5683'
c_grey= '#999999'

c_black_light= '#8075ae'
c_blue_light= '#6b98c8'
c_teal_light= '#8cc7cb'
c_green_light= '#76d7a2'
c_yellow_light= '#f8d6a1'
c_orange_light= '#f9bf88'
c_red_light= '#fd5d5a'
c_purple_light= '#d99ec2'
c_grey_light= '#eeeeee'
c_grey_extralight= '#fafafa'

c_black_dark= '#0d0823'
c_blue_dark= '#081d33'
c_teal_dark= '#104d52'
c_green_dark= '#0a4d28'
c_yellow_dark= '#9f6406'
c_orange_dark= '#894404'
c_red_dark= '#610200'
c_purple_dark= '#6f134c'
c_grey_dark= '#333333'


### Color Assignments ###

c_brand_color_main = c_black
c_brand_color_alt = c_blue
c_brand_color_acc = c_red

# Nobel-Spektrum category colors (Jewel = light mode). Single source: theme.SPECTRUM_LIGHT.
c_medicine   = "#B23A48"
c_physics    = "#2A4D9B"
c_chemistry  = "#C2682A"
c_economics  = "#6D3A9C"
c_literature = "#B08A1E"
c_peace      = "#1F7A5A"


c_pie1 = c_blue_light
c_pie2 = c_teal_light
c_pie3 = c_green_light
c_pie4 = c_yellow_light
c_pie5 = c_orange_light
c_pie6 = c_red_light
c_pie7 = c_purple_light
c_pie8 = c_black_light
c_pie0 = c_grey_light


c_plot_background = 'rgba(0,0,0,0)'   # transparent — redesign cards provide the surface
c_hoverlabel_bg = '#FFFFFF'

c_colorscale_palette = [c_blue, c_teal, c_green, c_yellow, c_orange, c_red, c_purple, c_black]
c_colorscale_palette_light = [c_blue_light, c_teal_light, c_green_light, c_yellow_light, c_orange_light, c_red_light, c_purple_light, c_black_light]

c_colorscale_red = [c_orange_light, c_red_dark]
c_colorscale_teal = [c_blue_light, c_teal_dark]
c_colorscale_contrast = [c_teal, c_red]


##################################################################################################
# Back-compat re-exports (PlotConfig/get_plot_configs and the filter renderers moved
# to plot_config.py / filters_ui.py; app.py and layouts.py still import them via cf.*)
##################################################################################################

from plot_config import PlotConfig, get_plot_configs      # noqa: E402,F401
from filters_ui import render_filter_standard, render_filter_nominations  # noqa: E402,F401
