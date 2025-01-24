# src/original_graph/graph_generator.py

from collections import deque
from typing import Optional, Tuple, Union

import networkx as nx
from numpy import ndarray

from config import Config
from utils import build_graph_nodes_and_edges, calculate_frame
from . import GraphAttrAgent
from .graph_node import GraphNode


class GraphGenerator:
    def __init__(self, attributes: GraphAttrAgent):
        GraphNode.initialize(attributes)

    def generate_network(self, frame_range: Optional[Tuple[int, int]] = None, regenerate_times: int = 100):
        frame_range = frame_range or Config.DEFAULT_FRAME_SIZE

        for _ in range(regenerate_times):
            nodes, edges = self._generate_graph_by_nodes_and_edges(frame_range)
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception("Failed to generate a original_graph within the specified attempts.")

        _synthetic_network = build_graph_nodes_and_edges(nodes, edges)
        frame = calculate_frame(graph=_synthetic_network, frame_range=frame_range)
        synthetic_network = self._filter_graph(_synthetic_network, frame)
        return synthetic_network

    def _generate_graph_by_nodes_and_edges(self, frame_range: Optional[Tuple[int, int]] = None):
        frame_range = frame_range or Config.DEFAULT_FRAME_SIZE

        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        __scaled_frame_range = (round(frame_range[0] * 1.1), round(frame_range[1] * 1.1))
        frame = calculate_frame(center_position=root_node.position, frame_range=__scaled_frame_range)

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

    def _filter_graph(self, graph, frame):
        nodes_to_keep = {node for node, pos in nx.get_node_attributes(graph, 'pos').items() if
                         self._within_frame(pos, frame)}
        filtered_graph = graph.subgraph(nodes_to_keep).copy()
        if filtered_graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(filtered_graph), key=len)
            filtered_graph = filtered_graph.subgraph(largest_cc).copy()
        return filtered_graph

    @staticmethod
    def _within_frame(position: Union[list, tuple, ndarray],
                      frame: Union[list[any, any], tuple[any, any], ndarray[any, any]]) -> bool:
        x, y = position
        return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]
