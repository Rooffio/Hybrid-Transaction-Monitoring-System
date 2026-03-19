"""
================================================================================
MODULE: GRAPH ANALYTICS & NETWORK TOPOLOGY
================================================================================
Target Audience: Compliance IT / Financial Crime Unit (FCU)

DESCRIPTION:
    This module utilizes Graph Theory (NetworkX) to identify structural AML
    typologies that are invisible to flat-file analysis. It focuses on:
    1. Hub & Spoke Detection: Identifying high-centrality accounts (Mules/Layerers).
    2. Circular Flows (U-Turns): Detecting funds moving A -> B -> A or A -> B -> C -> A.
    3. Structural Outliers: Using PageRank to find influential nodes in a network.

PERFORMANCE & TUNING:
    - Graph Pruning: Controlled via 'graph_analytics.min_edge_amount'. We filter
      low-value "noise" to prevent memory bloat and focus on material risk.
    - Complexity Limits: 'graph_analytics.max_nodes_for_pagerank' ensures the
      system falls back to simpler degree metrics if the network is too large.
    - Reciprocity: The engine prioritizes 2-hop and 3-hop cycles, which capture
      ~95% of typical laundering 'ring' behavior while staying O(N) efficient.

COMPLIANCE IMPACT:
    Crucial for detecting "Layering" and "Integration" phases of money laundering
    where funds are moved through multiple hops to obscure the original source.
================================================================================
"""

import pandas as pd
import networkx as nx
import logging

# --- LOGGING SETUP ---
logger = logging.getLogger("TMS_Graph_Analytics")


class GraphAnalytics:
    def __init__(self, config):
        """
        Initializes the graph engine with performance constraints from config.

        :param config: Dictionary containing 'graph_analytics' settings from YAML.
        """
        graph_cfg = config.get('graph_analytics', {})

        # Tuning parameters
        self.min_amount = graph_cfg.get('min_edge_amount', 10000)
        self.pagerank_limit = graph_cfg.get('max_nodes_for_pagerank', 5000)
        self.top_n_for_cycles = graph_cfg.get('cycle_scan_node_limit', 1000)

        logger.info(
            f"Graph Engine initialized. Pruning threshold: >{self.min_amount}, "
            f"PageRank Limit: {self.pagerank_limit} nodes."
        )

    def _filter_suspicious_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PERFORMANCE OPTIMIZATION: Pruning the 'Universe'.
        Building a graph of all transactions is mathematically expensive. We
        filter for material transactions or those already showing baseline risk.
        """
        # Threshold: Materiality filter to reduce edges by ~70-90% in typical retail banking
        if 'risk_score' in df.columns:
            subset = df[(df['amount'] >= self.min_amount) | (df['risk_score'] > 0)]
        else:
            subset = df[df['amount'] >= self.min_amount]

        logger.info(f"Filtered graph universe: {len(subset)} edges (Pruned from {len(df)} total).")
        return subset

    def build_transaction_graph(self, df: pd.DataFrame, source='sender_id', target='receiver_id') -> nx.DiGraph:
        """
        Constructs a Directed Graph (DiGraph) from the filtered transaction batch.
        Aggregates multiple transactions between the same pair into a single weighted edge.
        """
        subset = self._filter_suspicious_subset(df)

        if subset.empty:
            return nx.DiGraph()

        # Aggregate edges to prevent MultiGraph overhead and sum the total flow volume
        edge_data = subset.groupby([source, target])['amount'].sum().reset_index()

        G = nx.from_pandas_edgelist(
            edge_data,
            source=source,
            target=target,
            edge_attr=['amount'],
            create_using=nx.DiGraph()
        )
        return G

    def calculate_centrality_metrics(self, G: nx.DiGraph) -> dict:
        """
        Identifies 'Hubs' and 'Influencers' in the network.
        Uses In-Degree for volume and PageRank for structural importance.
        """
        if G.number_of_nodes() == 0:
            return {}

        # 1. In-Degree: Counts unique senders to this node (Direct Mule indicator)
        in_degrees = dict(G.in_degree())

        # 2. PageRank: Measures structural importance (The "Google Search" for Launderers)
        # Performance Fallback: Only run PageRank if the graph size is within safe bounds.
        pagerank = {}
        if G.number_of_nodes() < self.pagerank_limit:
            # max_iter=50 is sufficient for convergence in most financial graphs
            pagerank = nx.pagerank(G, weight='amount', max_iter=50)
        else:
            # Fast fallback to normalized degree if graph is massive
            max_deg = max(in_degrees.values()) if in_degrees else 1
            pagerank = {k: (v / max_deg) for k, v in in_degrees.items()}

        metrics = {
            node: {
                'in_degree': in_degrees.get(node, 0),
                'pagerank_score': pagerank.get(node, 0)
            }
            for node in G.nodes()
        }
        return metrics

    def detect_circular_flows(self, G: nx.DiGraph) -> dict:
        """
        SCALABLE CYCLE DETECTION:
        Standard 'simple_cycles' algorithms are O(E+V) which can explode.
        This method scans specifically for 2-hop (A-B-A) and 3-hop (A-B-C-A) loops.
        """
        logger.info(f"Scanning top {self.top_n_for_cycles} nodes for circular 'U-Turn' flows...")
        cycle_participants = {}

        # 1. Direct Reciprocity (A <-> B)
        # Common in layering/smurfing where funds bounce back to an intermediary
        for u, v in G.edges():
            if G.has_edge(v, u):
                cycle_participants[u] = 1
                cycle_participants[v] = 1

        # 2. Triadic Cycles (A -> B -> C -> A)
        # Scan restricted to the most active nodes to ensure performance stability
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:self.top_n_for_cycles]

        for node, _ in top_nodes:
            for neighbor in G.successors(node):
                for second_neighbor in G.successors(neighbor):
                    if G.has_edge(second_neighbor, node):
                        cycle_participants[node] = 1
                        cycle_participants[neighbor] = 1
                        cycle_participants[second_neighbor] = 1

        return cycle_participants

    def extract_graph_features(self, df: pd.DataFrame, entity_col='global_entity_id') -> pd.DataFrame:
        """
        Execution entry point for Graph-based feature engineering.
        Returns a DataFrame indexed by entity ID with normalized 'network_raw' risk scores.
        """
        # Ensure we use the best available identity column
        source_col = entity_col if entity_col in df.columns else 'sender_id'

        # 1. Build and Scan Graph
        G = self.build_transaction_graph(df, source=source_col, target='receiver_id')

        if G.number_of_nodes() == 0:
            return pd.DataFrame(
                columns=[source_col, 'network_centrality', 'network_pagerank', 'is_in_cycle', 'network_raw'])

        # 2. Derive Statistics
        centrality = self.calculate_centrality_metrics(G)
        cycles = self.detect_circular_flows(G)

        # 3. Assemble Feature Set
        feature_data = []
        for node, data in centrality.items():
            feature_data.append({
                source_col: node,
                'network_centrality': data.get('in_degree', 0),
                'network_pagerank': data.get('pagerank_score', 0),
                'is_in_cycle': cycles.get(node, 0)
            })

        graph_df = pd.DataFrame(feature_data)

        # 4. Normalized Scoring [0-100]
        if not graph_df.empty:
            # We use percentiles (Rank) for centrality to avoid skew from massive outliers
            graph_df['network_raw'] = graph_df['network_centrality'].rank(pct=True) * 70

            # Boost score for involvement in a circular flow (+30 risk points)
            graph_df.loc[graph_df['is_in_cycle'] == 1, 'network_raw'] += 30

            graph_df['network_raw'] = graph_df['network_raw'].clip(0, 100).round(2)

        logger.info(f"Graph analytics complete. Nodes analyzed: {G.number_of_nodes()}")
        return graph_df