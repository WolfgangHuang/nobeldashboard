"""
Facade for the former monolithic plotdatagenerator.py (5.5k lines), now split into:

    data.py             loading / cleaning / enrichment (runs once at import)
    filters.py          shared filter helpers
    figs_geography.py   geography figures
    figs_demography.py  demography figures
    figs_time.py        time / age figures
    figs_migration.py   migration figures
    figs_prizemoney.py  prize money figures
    network.py          nominations network + map
    overview_stats.py   overview KPIs

Everything is re-exported here so `import plotdatagenerator as pdg` (app.py) and
`from plotdatagenerator import ...` (config.py) keep working unchanged.

Running this file as a script regenerates the derived CSV files (done by
update_data.py after an API refresh).
"""

from data import *              # noqa: F401,F403
from data import data_path      # noqa: F401  (not star-exported by convention)
from filters import *           # noqa: F401,F403
from figs_geography import *    # noqa: F401,F403
from figs_demography import *   # noqa: F401,F403
from figs_time import *         # noqa: F401,F403
from figs_migration import *    # noqa: F401,F403
from figs_prizemoney import *   # noqa: F401,F403
from network import *           # noqa: F401,F403
from overview_stats import *    # noqa: F401,F403


##################################################################################################
# Save CSV Files
##################################################################################################

if __name__ == "__main__":
    import polars as pl

    # Save CSV files to disk

    df_laureates_enriched_full_clean.to_csv(data_path("df_laureates_enriched_full_clean.csv"), sep=';', encoding="UTF-8")
    print("df_laureates_enriched_full_clean.csv has been saved.")

    df_prizes_enriched_full_clean.to_csv(data_path("df_prizes_enriched_full_clean.csv"), sep=';', encoding="UTF-8")
    print("df_prizes_enriched_full_clean.csv has been saved.")

    df_laureates_enriched_redux_clean.to_csv(data_path("df_laureates_enriched_redux_clean.csv"), sep=';', encoding="UTF-8")
    print("df_laureates_enriched_redux_clean.csv has been saved.")

    df_prizes_enriched_redux_clean.to_csv(data_path("df_prizes_enriched_redux_clean.csv"), sep=';', encoding="UTF-8")
    print("df_prizes_enriched_redux_clean.csv has been saved.")

    # Regenerate the precomputed edge list from the current nominations. df_edges.csv
    # feeds the network, the nominations map and the live count at import time — if it
    # is not rewritten here, those widgets silently serve the state of the last export
    # (it once sat at 1901-1956 while nominations_full.csv already reached 1974).
    _edges_list, _skipped, _ = transform_to_edges(df_nominations)
    _df_edges_export = pl.DataFrame(_edges_list).select([
        "nomination_id", "year", "category", "motivation",
        "nominator_id", "nominator_name", "nominator_gender", "nominator_country",
        "nominee_id", "nominee_name", "nominee_gender", "nominee_country",
    ])
    _df_edges_export.write_csv(data_path("df_edges.csv"), separator=';')
    print(f"df_edges.csv has been saved ({len(_df_edges_export)} edges, "
          f"{_skipped} nominations skipped for missing nominator id).")
