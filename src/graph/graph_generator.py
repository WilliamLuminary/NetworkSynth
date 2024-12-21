# src/original_graph/graph_generator.py

from collections import deque

import networkx as nx
import numpy as np

from data.graph_data_agent import GraphDataAgent
from graph.graph_node import GraphNode
from config.base import BaseConfig
from utils.debug_utils import debugging


class GraphGenerator(BaseConfig):
    def __init__(self, graph_agent: GraphDataAgent):
        self.graph_agent = graph_agent
        self.graph_attributes = graph_agent.attributes

    @debugging
    def generate_network(self, frame_range: int = BaseConfig.DEFAULT_FRAME_RANGE, regenerate_times=100):
        for _ in range(regenerate_times):
            nodes, edges = self._generate_graph_by_nodes_and_edges(frame_range)
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception("Failed to generate a original_graph within the specified attempts.")

        _synthetic_network = self._build_network_from_nodes_and_edges(nodes, edges)
        frame = self._calculate_frame(graph=_synthetic_network, frame_range=frame_range)
        synthetic_network = self._filter_graph(_synthetic_network, frame)
        return synthetic_network, frame

    def _generate_graph_by_nodes_and_edges(self, frame_range: int = BaseConfig.DEFAULT_FRAME_RANGE):
        GraphNode.reset()
        GraphNode.initialize(self.graph_attributes)
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        frame = self._calculate_frame(position=root_node.position, frame_range=frame_range * 1.1)

        def __bfs(root):
            node_queue = deque([root])
            while node_queue:
                current_node = node_queue.popleft()
                if not self._within_frame(current_node.position, frame):
                    continue
                if current_node.generate_children():
                    for child in current_node.children:
                        if child != current_node:
                            node_set.add(child)
                            edge_set.add((current_node.position, child.position))
                            node_queue.append(child)

        __bfs(root_node)
        return node_set, edge_set

    @staticmethod
    def _build_network_from_nodes_and_edges(nodes, edges):
        graph = nx.Graph()
        position_map = {node.position: node for node in nodes}
        for node in nodes:
            graph.add_node(position_map[node.position].id, pos=node.position)

        for edge in edges:
            u, v = position_map[edge[0]].id, position_map[edge[1]].id
            graph.add_edge(u, v)
        return graph

    def _filter_graph(self, graph, frame):
        nodes_to_keep = {node for node, pos in nx.get_node_attributes(graph, 'pos').items() if
                         self._within_frame(pos, frame)}
        filtered_graph = graph.subgraph(nodes_to_keep).copy()
        if filtered_graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(filtered_graph), key=len)
            filtered_graph = filtered_graph.subgraph(largest_cc).copy()
        return filtered_graph

    @staticmethod
    def _calculate_frame(graph=None, position=None, frame_range=BaseConfig.DEFAULT_FRAME_RANGE):
        if graph is None and position is None:
            raise ValueError("Either synthetic_graph or position must be provided.")
        if graph is not None and position is not None:
            raise ValueError("Only one of synthetic_graph or position must be provided.")
        if graph is not None:
            positions = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
            center_x, center_y = positions[:, 0].mean(), positions[:, 1].mean()
        else:
            [center_x, center_y] = position
        half_range = frame_range / 2
        frame = [
            [round(center_x - half_range, 2), round(center_x + half_range, 2)],
            [round(center_y - half_range, 2), round(center_y + half_range, 2)]
        ]
        return frame

    @staticmethod
    def _within_frame(position, frame):
        x, y = position
        return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]
