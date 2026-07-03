# Nobel Prize Data Dashboard

A comprehensive interactive dashboard for exploring Nobel Prize data, built with Dash and Plotly. The dashboard provides visualizations covering laureate demographics, geographic distribution, historical trends, and nomination networks.

## Features

- Interactive visualizations with filtering by category, gender, and time range
- Geographic analysis: choropleth maps, city-level distributions, migration patterns
- Demographic breakdowns: gender, religion, ethnicity, age at award
- Time series analysis: prizes over time, publication-to-prize time gaps
- Nomination network graphs (based on historical nomination data with 50-year secrecy period)
- List generator with export functionality

## Project Structure

```
├── app.py                    # Main Dash application
├── config.py                 # Configuration and plot definitions
├── plotdatagenerator.py      # Data processing and plot generation
├── update_data.py            # API data update script
├── scrape_nominations.py     # Nomination archive scraper
├── df_laureates.csv          # Laureate data (from Nobel API)
├── nominations_full.csv      # Historical nomination data
└── [additional data files]   # Supporting datasets
```

## File Descriptions

### app.py

Main application entry point. Handles:

- Dash app initialization and layout (AppShell structure)
- Navigation and URL routing
- Callback definitions for plot updates and filter interactions
- Loading spinner management for lazy-loaded plots
- Section rendering (Overview, Geography, Demography, Time, etc.)

Key functions:
- `generate_loader_spinner()` - Creates placeholder spinners for plots
- `display_plot()` - Callback that renders plots when containers load
- `generate_plot_in_layout_standard()` - Builds widget layout for standard plots
- `generate_plot_in_layout_nominations()` - Builds widget layout for nomination plots
- `update_plot()` - Callback handling filter changes for standard plots

### config.py

Central configuration file containing:

- App settings (title, port, debug mode)
- Color palette definitions (category colors, chart colors)
- `PlotConfig` class for plot configuration management
- Plot configuration dictionary with all dashboard plots
- Filter panel rendering functions

Key components:
- `PlotConfig` class - Encapsulates all settings for a single plot (ID, title, generator function, filter options, styling)
- `render_filter_standard()` - Generates filter drawer for standard plots
- `render_filter_nominations()` - Generates filter drawer for nomination plots
- `get_plot_configs()` - Factory function returning all plot configurations

### plotdatagenerator.py

Data processing engine and plot generator library. Handles:

- Data import and cleaning (Pandas and Polars)
- Data enrichment (joining religion, ethnicity, movement data)
- Filter functions for data subsetting
- Plot generation functions for all visualization types

Key sections:

**Data Import (Lines 1-150)**
- Loads CSV/Excel files using lazy evaluation where possible
- Joins coordinate data for geographic visualizations
- Creates lookup tables for nominations

**Data Cleaning (Lines 150-450)**
- Standardizes country names
- Handles missing values
- Creates derived datasets (df_prizes from df_laureates)

**Filter Functions (Lines 500-1000)**
- `standard_filter()` - Pandas-based filtering for most plots
- `extended_filter()` - Polars-based filtering with lazy evaluation for list generator
- `filter_edges()` - Filters nomination network data

**Plot Generators (Lines 1000-5000)**

| Function | Description |
|----------|-------------|
| `generate_choroplethglobe()` | 3D globe with laureates per country |
| `generate_scattermapbox_cities()` | City-level scatter map |
| `generate_bubbles_perpopulation()` | Animated bubble chart (prizes vs population) |
| `generate_bar_percountry()` | Bar chart of prizes per country |
| `generate_3dsurface_pergender()` | 3D surface plot by gender |
| `generate_donut()` | Donut charts (gender, religion, ethnicity) |
| `generate_histogram_timegap()` | Publication-to-prize time gap histogram |
| `generate_scatterbox_age()` | Age at award scatter/box plot |
| `generate_heatmap_age()` | Age heatmap by category and decade |
| `generate_parcat_migration()` | Parallel categories for migration |
| `generate_network()` | NetworkX-based nomination graph |
| `generate_map_nominations()` | Geographic nomination flow map |

### update_data.py

Standalone script for updating laureate data from the Nobel Prize API. Designed for cron job execution.

Functions:
- `update_data()` - Main update orchestrator
- `validate_laureates_data()` - Validates API response before saving
- `run_plotdatagenerator()` - Triggers data processing after update
- `restart_app()` - Sends HUP signal to Gunicorn for graceful reload

### scrape_nominations.py

Web scraper for the Nobel Prize nomination archive (nobelprize.org).

Classes:
- `NobelNominationScraper` - Scraper with rate limiting and progress saving

Note: Nomination data has a 50-year secrecy period, so only historical nominations are available.

## Data Flow

```
                                    ┌─────────────────────┐
                                    │   Nobel Prize API   │
                                    └──────────┬──────────┘
                                               │
                                               v
┌─────────────────────┐            ┌─────────────────────┐
│  Nomination Archive │            │   update_data.py    │
│   (nobelprize.org)  │            │   (scheduled job)   │
└──────────┬──────────┘            └──────────┬──────────┘
           │                                  │
           v                                  v
┌─────────────────────┐            ┌─────────────────────┐
│scrape_nominations.py│            │  df_laureates.csv   │
└──────────┬──────────┘            └──────────┬──────────┘
           │                                  │
           v                                  │
┌─────────────────────┐                       │
│nominations_full.csv │                       │
└──────────┬──────────┘                       │
           │                                  │
           └────────────┬─────────────────────┘
                        │
                        v
           ┌─────────────────────┐
           │ plotdatagenerator.py│
           │                     │
           │  - Data cleaning    │
           │  - Enrichment       │
           │  - df_prizes        │
           │  - Polars DFs       │
           └──────────┬──────────┘
                      │
                      v
           ┌─────────────────────┐
           │     config.py       │
           │                     │
           │  - PlotConfig       │
           │  - Filter renderers │
           └──────────┬──────────┘
                      │
                      v
           ┌─────────────────────┐
           │      app.py         │
           │                     │
           │  - Dash callbacks   │
           │  - Layout rendering │
           │  - User interaction │
           └──────────┬──────────┘
                      │
                      v
           ┌─────────────────────┐
           │    Web Browser      │
           └─────────────────────┘
```

## Key Dependencies

- dash / dash-mantine-components - Web framework and UI components
- plotly - Interactive visualizations
- pandas - Data manipulation (laureate data)
- polars - High-performance data processing (nominations, list generator)
- networkx / pygraphviz - Network graph generation
- yfinance - Prize money currency conversion

## Performance Optimizations

The codebase uses several optimization techniques:

1. **Lazy Evaluation (Polars)**: Large datasets use `pl.scan_csv()` and LazyFrames to defer computation until results are needed.

2. **Query Optimization**: Filter chains in `extended_filter()` are built as query plans and executed in a single optimized pass.

3. **Pre-computed Lookups**: Hover data for animations uses dictionary lookups (O(1)) instead of DataFrame filtering (O(n)).

4. **Vectorized Operations**: String operations and data transformations use vectorized Polars/Pandas operations instead of row-wise apply.

## Running the Application

Development:
```bash
python app.py
```

Production (Gunicorn):
```bash
gunicorn -w 4 -b 0.0.0.0:8050 app:server
```

## Data Updates

To update laureate data from the API:
```bash
python update_data.py
```

This script:
1. Fetches current data from the Nobel Prize API
2. Validates the response
3. Saves raw data to CSV
4. Runs plotdatagenerator.py to process the data
5. Restarts the application (if running under Gunicorn)

## License

Data sourced from the official Nobel Prize API and nomination archive.
