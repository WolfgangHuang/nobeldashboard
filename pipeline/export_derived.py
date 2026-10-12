"""
Load and enrich all data exactly as the app does, then write the enriched tables
to data/derived/ (not in git).

update_data.py runs this after an API refresh before restarting the app: if the
new data breaks the loading code, this fails and the running app keeps its data.
The CSVs themselves are exports for analysis; the app does not read them.

Usage (from the project root):
    python -m pipeline.export_derived
"""

from nbldash import paths
from nbldash.data import (
    _edges_skipped,
    df_edges,
    df_laureates_enriched_full_clean,
    df_laureates_enriched_redux_clean,
    df_prizes_enriched_full_clean,
    df_prizes_enriched_redux_clean,
)


def main():
    paths.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    exports = {
        "df_laureates_enriched_full_clean.csv": df_laureates_enriched_full_clean,
        "df_prizes_enriched_full_clean.csv": df_prizes_enriched_full_clean,
        "df_laureates_enriched_redux_clean.csv": df_laureates_enriched_redux_clean,
        "df_prizes_enriched_redux_clean.csv": df_prizes_enriched_redux_clean,
    }
    for name, df in exports.items():
        df.to_csv(paths.derived(name), sep=';', encoding="UTF-8")
        print(f"{name} has been saved.")
    print(f"Nominations: {len(df_edges)} edges, "
          f"{_edges_skipped} nominations skipped for missing nominator id.")


if __name__ == "__main__":
    main()
