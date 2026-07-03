"""
Dead code: the pre-Cytoscape Plotly network path (kept for reference).

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
try:
    import pygraphviz  # noqa: F401  (optional; graphviz layouts fall back to spring)
except ImportError:
    pygraphviz = None
from data import df_nominations, lastyearincluded
from network import build_network_graph, clean_edges, filter_edges, transform_to_edges


def calculate_layout(G, LAYOUT_ALGORITHM, sfdp_k_value=0.3, sfdp_rf_value=1.0, sfdp_overlap="scale"):
    
    n_nodes = G.number_of_nodes()
    #print(f"Calculating layout for {n_nodes} nodes using {LAYOUT_ALGORITHM}...")
    
    # ========================================================================
    # PRE-COMPUTE NODE CATEGORIES (once, not per algorithm)
    # ========================================================================
    
    laureates = [n for n, d in G.nodes(data=True) if d.get('is_laureate', False)]
    both = [n for n, d in G.nodes(data=True) if d['type'] == 'both' and not d.get('is_laureate', False)]
    nominees = [n for n, d in G.nodes(data=True) if d['type'] == 'nominee' and not d.get('is_laureate', False)]
    nominators = [n for n, d in G.nodes(data=True) if d['type'] == 'nominator']
    
  
    # ========================================================================
    # LAYOUT ALGORITHMS
    # ========================================================================
    
    if LAYOUT_ALGORITHM == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    
    elif LAYOUT_ALGORITHM == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    elif LAYOUT_ALGORITHM == 'spectral':
        pos = nx.spectral_layout(G)
    
    elif LAYOUT_ALGORITHM == 'circular':
        pos = nx.circular_layout(G)
    
    elif LAYOUT_ALGORITHM == 'shell':
        shells = [laureates, both]
        other_nodes = [n for n in G.nodes() if n not in laureates and n not in both]
        shells.append(other_nodes)
        shells = [s for s in shells if s]
        pos = nx.shell_layout(G, nlist=shells)
    
    elif LAYOUT_ALGORITHM == 'spring_communities':
        # Better for clustered networks
        pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'hierarchical':
        # Layer by node type
        pos = {}
        layers = [laureates, both, nominees, nominators]
        y_positions = [3, 2, 1, 0]
        
        for layer, y_pos in zip(layers, y_positions):
            if not layer:
                continue
            spacing = 2
            x_offset = -(len(layer) - 1) / 2  # Center the layer
            for i, node in enumerate(layer):
                pos[node] = ((i + x_offset) * spacing, y_pos)
    
    elif LAYOUT_ALGORITHM == 'community':
        import networkx.algorithms.community as nx_comm
        
        #print("Detecting communities...")
        communities = list(nx_comm.louvain_communities(G.to_undirected()))
        #print(f"Detected {len(communities)} communities")
        
        pos = {}
        for i, community in enumerate(communities):
            # Position communities in circle
            angle = 2 * np.pi * i / len(communities)
            center_x = 15 * np.cos(angle)
            center_y = 15 * np.sin(angle)
            
            # Layout within community (much faster on small subgraphs)
            community_list = list(community)
            if len(community_list) > 1:
                subgraph = G.subgraph(community_list)
                # Reduce iterations for large communities
                iters = min(50, max(20, 100 // len(communities)))
                sub_pos = nx.spring_layout(subgraph, scale=4, iterations=iters, seed=42)
            else:
                sub_pos = {community_list[0]: (0, 0)}
            
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + center_x, y + center_y)
    
    elif LAYOUT_ALGORITHM == 'fruchterman_reingold':
        # Similar to spring but with different physics
        pos = nx.fruchterman_reingold_layout(G, k=3, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'multipartite':
        # Separate columns by node type
        subset_key = 'layer'
        
        # Assign layers (reuse pre-computed lists)
        for node in laureates:
            G.nodes[node][subset_key] = 0
        for node in both:
            G.nodes[node][subset_key] = 1
        for node in nominees:
            G.nodes[node][subset_key] = 2
        for node in nominators:
            G.nodes[node][subset_key] = 3
        
        pos = nx.multipartite_layout(G, subset_key=subset_key, scale=5)
    
    elif LAYOUT_ALGORITHM == 'graphviz_neato':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_osage':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='osage')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_patchwork':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='patchwork')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    elif LAYOUT_ALGORITHM == 'graphviz_fdp':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='fdp')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)

    
    elif LAYOUT_ALGORITHM == 'graphviz_sfdp':
        try:
            args_str = f'-GK={sfdp_k_value} -Grepulsiveforce={sfdp_rf_value} -Goverlap={sfdp_overlap}'
            # print(f"[SFDP] Using parameters: {args_str}")
            pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp', args=args_str)
        except Exception as e:
            print(f"[SFDP] Error: {e}")
            return calculate_layout(G, 'by_category')
   

    elif LAYOUT_ALGORITHM == 'graphviz_circo':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='circo')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'graphviz_twopi':
        # Force-directed with good clustering
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
        except Exception:
            print("pygraphviz not installed, falling back to spring")
            pos = nx.spring_layout(G, k=5, iterations=100, seed=42)
    
    elif LAYOUT_ALGORITHM == 'by_category':
        # Group by scientific category - FAST for large graphs
        #print("Grouping by category...")
        
        # Group nodes by category (reuse node data)
        categories = {}
        for node, data in G.nodes(data=True):
            cat = data.get('main_category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(node)
        
        # print(f"Found {len(categories)} categories")
        
        pos = {}
        num_cats = len(categories)
        
        for i, (cat, nodes) in enumerate(categories.items()):
            # Calculate center for this category
            angle = 2 * np.pi * i / num_cats
            center_x = 20 * np.cos(angle)
            center_y = 20 * np.sin(angle)
            
            # Layout within category (fast on subgraphs)
            if len(nodes) > 1:
                subgraph = G.subgraph(nodes)
                # Scale iterations with subgraph size
                iters = min(50, max(20, 1000 // len(nodes)))
                sub_pos = nx.spring_layout(subgraph, scale=5, k=2, iterations=iters, seed=42)
            else:
                sub_pos = {nodes[0]: (0, 0)}
            
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + center_x, y + center_y)
    
    elif LAYOUT_ALGORITHM == 'bipartite':
        # Only works if graph is actually bipartite
        nominator_nodes = set(nominators + both)
        nominee_nodes = set(nominees + both)
        
        # Set bipartite attribute
        for node in nominator_nodes:
            G.nodes[node]['bipartite'] = 0
        for node in nominee_nodes:
            G.nodes[node]['bipartite'] = 1
        
        pos = nx.bipartite_layout(G, nominator_nodes, scale=5)
    
    elif LAYOUT_ALGORITHM == 'random':
        pos = nx.random_layout(G, seed=42)
    
    else:
        print(f"Unknown layout algorithm: {LAYOUT_ALGORITHM}")
        print("Falling back to 'by_category' (good default for large graphs)")
        return calculate_layout(G, 'by_category')
    
    #print(f"Layout calculation complete!")
    return pos


def create_network_figure(G, pos, df_edges, highlighted_person_id=None, debug=False):
    """
    Create Plotly network figure with optimized data preparation
    """
    
    if debug:
        print(f"\n[DEBUG] create_network_figure called (highlighted_person_id={highlighted_person_id})")
    
    import time
    start_time = time.time()
    
    # Remove nodes from pos that are not in G
    pos = {k: v for k, v in pos.items() if k in G.nodes}
    
    # ========================================================================
    # STEP 1: Pre-build lookup structures (CRITICAL for performance)
    # ========================================================================
    
    if debug:
        print("Building lookup structures...")
    
    # Build node connections lookup (instead of iterating edges for each node)
    node_out_edges = defaultdict(list)  # node -> list of (target, edge_data)
    node_in_edges = defaultdict(list)   # node -> list of (source, edge_data)
    
    # Build nomination groups lookup
    edges_by_nomination = defaultdict(list)
    nomination_data = {}  # Store first edge data per nomination
    
    for u, v, data in G.edges(data=True):
        nomination_id = data['nomination_id']
        
        # Cache connections
        node_out_edges[u].append((v, data))
        node_in_edges[v].append((u, data))
        
        # Cache nomination grouping
        edges_by_nomination[nomination_id].append((u, v, data))
        
        # Store first edge data for hover info
        if nomination_id not in nomination_data:
            nomination_data[nomination_id] = data
    
    if debug:
        print(f"  Built lookups in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 2: Determine highlighted nominations (if any)
    # ========================================================================
    
    highlighted_nominations = set()
    if highlighted_person_id is not None:
        # Use cached connections instead of iterating all edges
        for target, data in node_out_edges[highlighted_person_id]:
            highlighted_nominations.add(data['nomination_id'])
        for source, data in node_in_edges[highlighted_person_id]:
            highlighted_nominations.add(data['nomination_id'])
    
   # ========================================================================
    # STEP 3: Create edge traces (batch processing)
    # ========================================================================

    # ============================================================================
    # VISUAL CONFIGURATION - Adjust styling here
    # ============================================================================

    # Opacity configuration (applies to both nodes and edges)
    OPACITY_CONFIG = {
        'base': 0.8,              # No highlighting active
        'highlighted': 1.0,        # Highlighted nodes/edges
        'not_highlighted': 0.4    # Dimmed nodes/edges when highlighting active
    }

    # Highlighting colors
    HIGHLIGHT_CONFIG = {
        'color': cf.c_brand_color_acc,
        'non_highlighted_color': 'darkgrey'
    }

    # ============================================================================

    if debug:
        print("Creating edge traces...")

    # Category colors for edges
    category_colors_edges = {
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Physiology or Medicine': cf.c_medicine,
        'Medicine': cf.c_medicine,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace,
        'Economic Sciences': cf.c_economics,
        'Economics': cf.c_economics,
    }

    edge_traces = []

    for nomination_id, edges in edges_by_nomination.items():
        edge_x = []
        edge_y = []
        
        for u, v, data in edges:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        if not edge_x:
            continue
        
        # Use cached nomination data
        first_edge_data = nomination_data[nomination_id]
        co_nominees_str = ", ".join(first_edge_data['co_nominees'])
        
        # Handle None in motivation
        motivation = first_edge_data['motivation']
        motivation_text = (
            "No motivation provided" if motivation is None 
            else (str(motivation)[:100] + ("..." if len(str(motivation)) > 100 else ""))
        )
        
        # Determine styling
        is_highlighted = nomination_id in highlighted_nominations
        is_group = first_edge_data['is_group_nomination']
        edge_category = first_edge_data['category']
        
        # Get category color
        base_edge_color = category_colors_edges.get(edge_category, cf.c_grey)
        
        if is_highlighted:
            line_width = 5
            opacity = OPACITY_CONFIG['highlighted']
            line_color = HIGHLIGHT_CONFIG['color']
        elif highlighted_person_id is not None:
            line_width = 1
            opacity = OPACITY_CONFIG['not_highlighted']
            line_color = HIGHLIGHT_CONFIG['non_highlighted_color']
        else:
            line_width = 3 if is_group else 2
            opacity = OPACITY_CONFIG['base']
            line_color = base_edge_color
        
        hover_text = (
            f"<b>Nomination ID: {nomination_id}</b><br>"
            f"Year: {first_edge_data['year']}<br>"
            f"Category: {first_edge_data['category']}<br>"
            f"Nominees: {co_nominees_str}<br>"
            f"Motivation: {motivation_text}"
        )
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=line_width, color=line_color),
            hoverinfo='text',
            text=hover_text,
            opacity=opacity,
            showlegend=False,
            meta={
                'type': 'edge',
                'nomination_id': nomination_id,
                'category': edge_category,
                'connected_persons': list(set([u for u, v, d in edges] + [v for u, v, d in edges]))
            }
        )
        edge_traces.append(edge_trace)

    if debug:
        print(f"  Created {len(edge_traces)} edge traces in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 4: Pre-build node hover texts (HUGE optimization)
    # ========================================================================
    
    if debug:
        print("Building node hover texts...")
    
    # Category colors
    category_colors = {
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Physiology or Medicine': cf.c_medicine,
        'Medicine': cf.c_medicine,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace,
        'Economic Sciences': cf.c_economics,
        'Economics': cf.c_economics,
    }
    
    # Build hover texts for ALL nodes at once
    node_hover_cache = {}
    
    for node, node_data in G.nodes(data=True):
        if node not in pos or 'name' not in node_data:
            continue
        
        # Use cached connections (MUCH faster than iterating all edges)
        nominated_list = []
        for target, data in node_out_edges[node]:
            if target in G.nodes and 'name' in G.nodes[target]:
                nominated_list.append({
                    'name': G.nodes[target]['name'],
                    'year': data['year'],
                    'category': data['category']
                })
        
        was_nominated_list = []
        for source, data in node_in_edges[node]:
            if source in G.nodes and 'name' in G.nodes[source]:
                was_nominated_list.append({
                    'name': G.nodes[source]['name'],
                    'year': data['year'],
                    'category': data['category']
                })
        
        # Build hover text
        hover_lines = [
            f"<b>{node_data['name']}</b>",
            f"Country: {node_data['country']}",
            f"Category: {node_data.get('main_category', 'Unknown')}",
        ]
        
        # Add status indicator for primary vs secondary nodes
        is_primary = node_data.get('is_primary', True)
        if not is_primary:
            hover_lines.append("<i>(Connected via filter match)</i>")
        
        hover_lines.append("")  # Empty line
        
        if node_data.get('is_laureate', False):
            hover_lines.insert(3, "Status: Nobel Laureate ⭐")
        
        if nominated_list:
            hover_lines.append("<b>Nominated the following persons:</b>")
            for nom in nominated_list[:5]:
                hover_lines.append(f"  - {nom['name']}, {nom['year']}, {nom['category']}")
            if len(nominated_list) > 5:
                hover_lines.append(f"  ... and {len(nominated_list) - 5} more")
            hover_lines.append("")
        
        if was_nominated_list:
            hover_lines.append("<b>Was nominated by the following persons:</b>")
            for nom in was_nominated_list[:5]:
                hover_lines.append(f"  - {nom['name']}, {nom['year']}, {nom['category']}")
            if len(was_nominated_list) > 5:
                hover_lines.append(f"  ... and {len(was_nominated_list) - 5} more")
        
        hover_lines.append("")
        hover_lines.append("<i>Click to highlight</i>")
        
        node_hover_cache[node] = "<br>".join(hover_lines)
    
    if debug:
        print(f"  Built {len(node_hover_cache)} hover texts in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 5: Create node traces (using cached data)
    # ========================================================================

    # Node styling configuration by type
    NODE_STYLES = {
        'laureate': {
            'symbol': 'circle',
            'base_color': None,  # None = use category color
            'size': 12,
            'line_width': 1,
            'line_color': 'black'
        },
        'both': {
            'symbol': 'circle',
            'base_color': 'grey',
            'size': 10,
            'line_width': 1,
            'line_color': 'black'
        },
        'nominator': {
            'symbol': 'circle',
            'base_color': 'lightgrey',
            'size': 10,
            'line_width': 1,
            'line_color': 'grey'
        },
        'nominee': {
            'symbol': 'circle',
            'base_color': 'grey',
            'size': 10,
            'line_width': 1,
            'line_color': 'lightgrey'
        }
    }

    # Note: OPACITY_CONFIG and HIGHLIGHT_CONFIG are defined in STEP 3

    if debug:
        print("Creating node traces...")

    # Create nodes_by_symbol dict dynamically from NODE_STYLES
    # Now with separate tracking for primary and secondary nodes
    nodes_by_symbol = {}
    for node_type, style in NODE_STYLES.items():
        symbol = style['symbol']
        if symbol not in nodes_by_symbol:
            nodes_by_symbol[symbol] = {
                'x': [], 'y': [], 'text': [], 'color': [], 'size': [], 'ids': [], 'is_primary': []
            }

    for node, node_data in G.nodes(data=True):
        if node not in pos or node not in node_hover_cache:
            continue
        
        x, y = pos[node]
        
        # Check if this is a primary node (matched the name filter directly)
        is_primary = node_data.get('is_primary', True)
        
        # Determine node type
        if node_data.get('is_laureate', False):
            node_type = 'laureate'
        elif node_data['type'] == 'both':
            node_type = 'both'
        elif node_data['type'] == 'nominator':
            node_type = 'nominator'
        else:
            node_type = 'nominee'
        
        # Get style configuration
        style = NODE_STYLES[node_type]
        
        # Determine color
        main_category = node_data.get('main_category', 'Unknown')
        is_highlighted = (node == highlighted_person_id)
        
        if is_highlighted:
            color = HIGHLIGHT_CONFIG['color']
        elif highlighted_person_id is not None:
            color = HIGHLIGHT_CONFIG['non_highlighted_color']
        elif style['base_color']:
            color = style['base_color']
        else:
            color = category_colors.get(main_category, cf.c_grey)
        
        # Adjust size for secondary nodes (smaller)
        node_size = style['size'] if is_primary else style['size'] * 0.7
        
        # Add to corresponding symbol dict
        symbol = style['symbol']
        nodes_by_symbol[symbol]['x'].append(x)
        nodes_by_symbol[symbol]['y'].append(y)
        nodes_by_symbol[symbol]['text'].append(node_hover_cache[node])
        nodes_by_symbol[symbol]['color'].append(color)
        nodes_by_symbol[symbol]['size'].append(node_size)
        nodes_by_symbol[symbol]['ids'].append(node)
        nodes_by_symbol[symbol]['is_primary'].append(is_primary)

    # Create separate traces for each symbol
    node_traces = []
    for symbol, data in nodes_by_symbol.items():
        if len(data['x']) > 0:
            # Get line config for this symbol
            line_config = next((s for s in NODE_STYLES.values() if s['symbol'] == symbol), NODE_STYLES['both'])
            
            # Determine opacity
            if highlighted_person_id is None:
                opacity = OPACITY_CONFIG['base']
            else:
                opacity = OPACITY_CONFIG['highlighted']
            
            trace = go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                hoverinfo='text',
                text=data['text'],
                customdata=data['ids'],
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    symbol=symbol,
                    opacity=opacity,
                    line=dict(
                        width=line_config['line_width'], 
                        color=line_config['line_color'] if line_config['line_color'] else data['color']
                    )
                ),
                showlegend=False,
                meta={
                    'type': 'node',
                    'symbol': symbol
                }
            )
            node_traces.append(trace)

    if debug:
        print(f"  Created {len(node_traces)} node traces in {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # STEP 6: Create figure
    # ========================================================================
    
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Add legend entries for node types (invisible dummy traces)
    legend_traces = []
    
    # Nominator (only nominator, not nominee)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='lightgrey', symbol='circle', 
                   line=dict(width=1, color='grey')),
        name='Nominator',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Nominee (only nominee, not nominator)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='grey', symbol='circle',
                   line=dict(width=1, color='lightgrey')),
        name='Nominee',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Both (nominee and nominator)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='grey', symbol='circle',
                   line=dict(width=1, color='black')),
        name='Nominee & Nominator',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Laureate (colored by category)
    legend_traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle',
                   line=dict(width=1, color='black')),
        name='Laureate',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add category legend entries
    category_colors = {
        'Medicine': cf.c_medicine,
        'Physics': cf.c_physics,
        'Chemistry': cf.c_chemistry,
        'Economic Sciences': cf.c_economics,
        'Literature': cf.c_literature,
        'Peace': cf.c_peace
    }
    
    for category, color in category_colors.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=2, color=color),
            name=category,
            showlegend=True,
            hoverinfo='skip'
        ))
    
    fig.add_traces(legend_traces)

    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Legend",
                        method="relayout",
                        args=[{"showlegend": True}],
                        args2=[{"showlegend": False}]
                    )
                ],
                x=0.01,
                y=0.01,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=cf.c_grey,
                borderwidth=1
            )
        ],
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=cf.c_plot_background,
        template='nbl_light',
        height=800,
        font=dict(family='IBM Plex Sans, sans-serif', size=12, color=cf.c_brand_color_main),
        hoverlabel=dict(bgcolor=cf.c_plot_background, font_size=12, font_family="IBM Plex Sans"),
    )


    
    if debug:
        print(f"\n[DEBUG] Total time: {time.time() - start_time:.2f}s")
        print(f"  Edge traces: {len(edge_traces)}")
        print(f"  Node traces: {len(node_traces)}")
    
    return fig


def generate_network(data=df_nominations,
                     algorithm="graphviz_sfdp", sfdp_k_value=1.0, sfdp_rf_value=1.0, sfdp_overlap="prism",
                     categories="all", timerange_nomination=[1901, lastyearincluded-49], timerange_field="nomination",
                     nominator_name="", nominator_search_mode="all", nominator_gender="all", nominator_country="all", nominator_islaureate=False, 
                     nominee_name="", nominee_search_mode="all", nominee_gender="female", nominee_country="all", nominee_islaureate=False, highlighted_person_id=None):

    # Step 1: Create Polars DataFrame with all edges

    # print(f"\n[ALGORITHM] {algorithm}")

    func_start_time = time.time()
    edges_list, skipped_nominations, skipped_nominations_list = transform_to_edges(data)
    df_edges = clean_edges(edges_list)
    # print(f"[TIME] Create Edges: {time.time() - func_start_time:.2f}s")


    # Step 2: Filter
    func_start_time = time.time()
    df_edges = filter_edges(df_edges, categories, timerange_nomination, timerange_field, nominator_name, nominator_search_mode, nominator_gender, nominator_country, nominator_islaureate, nominee_name, nominee_search_mode, nominee_gender, nominee_country, nominee_islaureate)
    # print(f"[TIME] Filter: {time.time() - func_start_time:.2f}s")
    # print(f"[SIZE] Rows: {len(df_edges)}")

    # Step 3: Build network graph
    func_start_time = time.time()
    G = build_network_graph(df_edges, data)
    # print(f"[TIME] Build network graph: {time.time() - func_start_time:.2f}s")
    # print(f"[SIZE] Edges: {G.number_of_edges()}")

    # Step 4: Calculate layout
    func_start_time = time.time()
    pos = calculate_layout(G, algorithm, sfdp_k_value, sfdp_rf_value, sfdp_overlap,)
    # print(f"[TIME] Calculate layout: {time.time() - func_start_time:.2f}s")

    # Step 5: Generate plot
    func_start_time = time.time()
    initial_fig = create_network_figure(G, pos, df_edges, highlighted_person_id, debug=False)
    # print(f"[TIME] Generate plot: {time.time() - func_start_time:.2f}s")

    return initial_fig, G
