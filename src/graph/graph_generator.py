# src/graph/graph_generator.py

import numpy as np
import networkx as nx
from collections import deque
from utils.debug_utils import debugging
from .graph_node import GraphNode


class GraphGenerator:
    def __init__(self, graph_attributes, config):
        self.graph_attributes = graph_attributes
        self.config = config

    @debugging
    def generate_graph(self, frame_range=None, regenerate_times=100):
        if frame_range is None:
            frame_range = self.config.DEFAULT_FRAME_RANGE
        for _ in range(regenerate_times):
            nodes, edges = self._generate_nodes_and_edges(frame_range)
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception("Failed to generate a graph within the specified attempts.")

        graph = self._build_graph_from_nodes_and_edges(nodes, edges)
        frame = self._calculate_frame(graph, frame_range)
        synthetic_graph = self._filter_graph(graph, frame)
        return synthetic_graph, frame

    def _generate_nodes_and_edges(self, frame_range):
        GraphNode.reset()
        GraphNode.initialize(self.graph_attributes, self.config)
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        frame = self._calculate_frame(root_node.position[0], root_node.position[1], frame_range)

        def bfs(root):
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

        bfs(root_node)
        return node_set, edge_set

    def _build_graph_from_nodes_and_edges(self, nodes, edges):
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

    def _calculate_frame(self, graph_or_x, y=None, frame_range=None):
        if frame_range is None:
            frame_range = self.config.DEFAULT_FRAME_RANGE
        if isinstance(graph_or_x, nx.Graph):
            positions = np.array(list(nx.get_node_attributes(graph_or_x, 'pos').values()))
            center_x, center_y = positions[:, 0].mean(), positions[:, 1].mean()
        else:
            center_x, center_y = graph_or_x, y
        half_range = frame_range / 2
        frame = [
            [round(center_x - half_range, 2), round(center_x + half_range, 2)],
            [round(center_y - half_range, 2), round(center_y + half_range, 2)]
        ]
        return frame

    def _within_frame(self, position, frame):
        x, y = position
        return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]
