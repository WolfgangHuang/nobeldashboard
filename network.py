"""
Nominations network: edge building/filtering, NetworkX graph, cytoscape elements, map.

Extracted from plotdatagenerator.py (facade re-exports everything).
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
import config as cf
import theme as th
import networkx as nx
import time
from collections import defaultdict
from data import data_path, df_edges, df_nominations, lastyearincluded


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

    df_match_edges_country = pl.read_csv(data_path('edges_country_match.csv'), separator=';', encoding='utf8')
    df_coordinates = pl.read_csv(data_path('countries_with_coordinates.csv'), separator=';', encoding='utf8')

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


def _compute_laureate_ids(df_noms):
    """IDs of everyone who ever won a prize (any nominee_*_awarded_prizes set)."""
    ids = set()
    for i in range(1, max_nominees + 1):
        id_col = f'nominee_{i}_id'
        award_col = f'nominee_{i}_awarded_prizes'
        if id_col in df_noms.columns and award_col in df_noms.columns:
            laureates = (
                df_noms
                .filter(pl.col(award_col).is_not_null() & (pl.col(award_col) != ''))
                .select(pl.col(id_col).cast(pl.Int64, strict=False))
                .unique()
            )
            ids.update(v for v in laureates[id_col].to_list() if v is not None)
    return ids


_NOMINATIONS_DEFAULT = df_nominations


_LAUREATE_IDS = _compute_laureate_ids(df_nominations)


def build_network_graph(df_edges=df_edges, df_nominations=df_nominations):

    # Use MultiDiGraph to allow multiple edges between same nodes
    # (same person can nominate another person in different years)
    G = nx.MultiDiGraph()

    # ========================================================================
    # STEP 1: Laureate IDs (precomputed for the module-level frame)
    # ========================================================================

    if df_nominations is _NOMINATIONS_DEFAULT:
        laureate_ids = _LAUREATE_IDS
    else:
        laureate_ids = _compute_laureate_ids(df_nominations)

    # ========================================================================
    # STEP 2: One pass over the edge rows builds person data, category counts,
    # graph edges and co-nominee groups (was four separate iterrows loops)
    # ========================================================================

    all_people = {}

    # Track which persons are primary (matched the name filter directly)
    primary_person_ids = set()

    # Count category appearances for each person
    category_counts = defaultdict(lambda: defaultdict(int))

    edges_to_add = []

    # Group nominees by nomination_id (for co-nominee info)
    nomination_groups = defaultdict(list)

    # Convert to pandas for fast iteration (Polars iter_rows is slow)
    df_edges_pd = df_edges.to_pandas()

    # Check if primary marker columns exist
    has_primary_markers = 'is_primary_nominator' in df_edges_pd.columns

    for row in df_edges_pd.itertuples(index=False):
        nominator_id = row.nominator_id
        nominee_id = row.nominee_id
        category = row.category

        # Track primary persons
        if has_primary_markers:
            if row.is_primary_nominator:
                primary_person_ids.add(nominator_id)
            if row.is_primary_nominee:
                primary_person_ids.add(nominee_id)

        # Add/update nominator
        person = all_people.get(nominator_id)
        if person is None:
            all_people[nominator_id] = {
                'name': row.nominator_name,
                'country': row.nominator_country,
                'type': 'nominator',
                'categories': {category},
                'is_laureate': nominator_id in laureate_ids
            }
        else:
            person['categories'].add(category)

        # Add/update nominee
        person = all_people.get(nominee_id)
        if person is None:
            all_people[nominee_id] = {
                'name': row.nominee_name,
                'country': row.nominee_country,
                'type': 'nominee',
                'categories': {category},
                'is_laureate': nominee_id in laureate_ids
            }
        else:
            person['categories'].add(category)
            # Update type if person is both nominator and nominee
            if person['type'] == 'nominator':
                person['type'] = 'both'

        category_counts[nominator_id][category] += 1
        category_counts[nominee_id][category] += 1

        edges_to_add.append((
            nominator_id,
            nominee_id,
            {
                'nomination_id': row.nomination_id,
                'year': row.year,
                'category': category,
                'motivation': row.motivation
            }
        ))

        nomination_groups[row.nomination_id].append(row.nominee_name)

    # ========================================================================
    # STEP 3: Determine main category and finalize person data
    # ========================================================================

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
    # STEP 4: Add nodes + edges to graph (batch operations)
    # ========================================================================

    G.add_nodes_from(all_people.items())
    G.add_edges_from(edges_to_add)

    # ========================================================================
    # STEP 5: Add co-nominee information
    # ========================================================================

    for u, v, data in G.edges(data=True):
        nomination_id = data['nomination_id']
        data['co_nominees'] = nomination_groups[nomination_id]
        data['is_group_nomination'] = len(nomination_groups[nomination_id]) > 1

    return G


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
    if data is df_nominations:
        # Reuse the module-level precomputed df_edges (edge list + country/coordinate/
        # prize joins, done once at import) — the same source the map path and the
        # live nomination count already use. Rebuilding it here from the raw
        # nominations (transform_to_edges + clean_edges, incl. two CSV reads from
        # disk) cost ~400 ms per callback; filtering the precomputed frame is ~20 ms.
        df_edges_local = df_edges
    else:
        edges_list, _skipped, _skipped_list = transform_to_edges(data)
        df_edges_local = clean_edges(edges_list)
    df_edges_local = filter_edges(df_edges_local, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    G = build_network_graph(df_edges_local, data)
    elements = graph_to_cytoscape_elements(G)
    return elements, G


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
