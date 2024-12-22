# src/original_graph/graph_postprocessor.py

import networkx as nx
from scipy.spatial.distance import euclidean

from config.config import Config
from handlers.data_agent import DataAgent


class GraphPostProcessor(Config):
    def __init__(self, synthetic_graph, data_agent: DataAgent):
        self.synthetic_graph = synthetic_graph
        self.original_graph_attributes = data_agent.attributes

    def remove_nodes_and_edges_by_degree_distribution(self):
        target_avg_degree = self.original_graph_attributes.avg_degree
        graph = self.synthetic_graph.copy()

        while 2 * graph.number_of_edges() / graph.number_of_nodes() > 1.1 * target_avg_degree:
            highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]
            neighbors = list(graph.neighbors(highest_degree_node))
            if neighbors:
                graph.remove_edge(highest_degree_node, neighbors[0])
            graph = self._keep_largest_connected_component(graph)
        self.synthetic_graph = graph

    def assign_weights(self):
        [_, map_length_to_weight, length_bins, weight_baskets, edge_weights] = self.original_graph_attributes.mapping_params
        edge_lengths = [euclidean(self.synthetic_graph.nodes[u]['pos'], self.synthetic_graph.nodes[v]['pos']) for u, v
                        in
                        self.synthetic_graph.edges()]
        weights = [map_length_to_weight(length, length_bins, weight_baskets, edge_weights) for length in edge_lengths]
        for (u, v), weight in zip(self.synthetic_graph.edges(), weights):
            self.synthetic_graph[u][v]['weight'] = weight

    @staticmethod
    def _keep_largest_connected_component(graph):
        if graph.number_of_nodes() == 0:
            return graph
        largest_cc = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_cc).copy()
