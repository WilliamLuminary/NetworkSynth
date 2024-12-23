# src/original_graph/graph_postprocessor.py

import networkx as nx

from config import Config
from handlers import DataAgent


class GraphPostProcessor(Config):
    def __init__(self, synthetic_graph, data_agent: DataAgent):
        self.synthetic_graph = synthetic_graph
        self.data_agent = data_agent

    def trim_graph(self):
        """
        Trim the synthetic graph to have an average degree close to the original graph.
        """
        _graph = self.synthetic_graph
        target_avg_degree = self.data_agent.attributes.avg_degree

        while 2 * _graph.number_of_edges() / _graph.number_of_nodes() > 1.1 * target_avg_degree:
            highest_degree_node = max(_graph.degree, key=lambda x: x[1])[0]
            neighbors = list(_graph.neighbors(highest_degree_node))
            if neighbors:
                _graph.remove_edge(highest_degree_node, neighbors[0])
            _graph = self._keep_largest_connected_component(_graph)

        self.synthetic_graph = _graph

    def assign_weights(self):
        self.data_agent.map_handler.assign_weights(self.synthetic_graph)

    @staticmethod
    def _keep_largest_connected_component(graph):
        if graph.number_of_nodes() == 0:
            return graph
        largest_cc = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_cc).copy()
