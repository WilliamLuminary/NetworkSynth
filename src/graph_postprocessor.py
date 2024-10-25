# graph_postprocessor.py

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean
from utils.debug_utils import debugging


class GraphPostProcessor:
    def __init__(self, graph, original_metrics):
        self.graph = graph
        self.original_metrics = original_metrics

    @debugging
    def adjust_degree_distribution(self):
        target_avg_degree = self.original_metrics['avg_degree']
        graph = self.graph.copy()

        while 2 * graph.number_of_edges() / graph.number_of_nodes() > 1.1 * target_avg_degree:
            highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]
            neighbors = list(graph.neighbors(highest_degree_node))
            if neighbors:
                graph.remove_edge(highest_degree_node, neighbors[0])
            graph = self._keep_largest_connected_component(graph)
        self.graph = graph

    def assign_weights(self, map_length_to_weight, length_bins, weight_baskets, y):
        edge_lengths = [euclidean(self.graph.nodes[u]['pos'], self.graph.nodes[v]['pos']) for u, v in
                        self.graph.edges()]
        weights = [map_length_to_weight(length, length_bins, weight_baskets, y) for length in edge_lengths]
        for (u, v), weight in zip(self.graph.edges(), weights):
            self.graph[u][v]['weight'] = weight

    def _keep_largest_connected_component(self, graph):
        if graph.number_of_nodes() == 0:
            return graph
        largest_cc = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_cc).copy()
