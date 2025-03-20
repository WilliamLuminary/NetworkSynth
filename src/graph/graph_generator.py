# src/graph/graph_generator.py

from typing import Optional, Tuple, Union

import networkx as nx
from numpy import ndarray

from config import BaseConfig
from utils import build_graph, calculate_frame, largest_connected_component
from ._graph_node import GraphNode


class GraphGenerator:
    from handlers import AttributesCalculator
    def __init__(self, attributes_calculator: AttributesCalculator):
        GraphNode.initialize(attributes_calculator)

    def generate_network(self, frame_range: Optional[Tuple[int, int]] = None, regenerate_times: int = 100):
        frame_range = frame_range or BaseConfig.DEFAULT_FRAME_SIZE

        for _ in range(regenerate_times):
            nodes, edges = self._bfs_network(frame_range)
            if nodes and len(nodes) > 100:
                break
        else:
            raise Exception("Failed to generate a original_network within the specified attempts.")

        _synthetic_network = build_graph(nodes, edges, arg_type='edge_list')
        frame = calculate_frame(graph=_synthetic_network, frame_range=frame_range)
        synthetic_network = _filter_graph(_synthetic_network, frame)
        return synthetic_network

    @staticmethod
    def _bfs_network(frame_range: Optional[Tuple[int, int]] = None) -> Tuple[set, set]:
        frame_range = frame_range or BaseConfig.DEFAULT_FRAME_SIZE

        GraphNode.reset()
        root_node = GraphNode((0, 0))
        node_set, edge_set = {root_node}, set()
        scaled_frame_range = (round(frame_range[0] * 1.1), round(frame_range[1] * 1.1))
        frame = calculate_frame(center_position=root_node.position, frame_range=scaled_frame_range)

        from collections import deque
        node_queue = deque([root_node])
        while node_queue:
            current_node = node_queue.popleft()
            if not _within_frame(current_node.position, frame):
                continue
            if current_node.generate_children():
                for child in current_node.children:
                    if child != current_node:
                        node_set.add(child)
                        edge_set.add((current_node.position, child.position))
                        node_queue.append(child)
        return node_set, edge_set


def _within_frame(position: Union[list, tuple, ndarray],
                  frame: Union[list[any, any], tuple[any, any], ndarray[any, any]]) -> bool:
    x, y = position
    return frame[0][0] <= x <= frame[0][1] and frame[1][0] <= y <= frame[1][1]


def _filter_graph(graph: nx.Graph, frame: [list[any, any], tuple[any, any]]) -> nx.Graph:
    nodes_to_keep = {node for node, pos in nx.get_node_attributes(graph, 'pos').items() if
                     _within_frame(pos, frame)}
    filtered_fraction = graph.subgraph(nodes_to_keep)
    return largest_connected_component(filtered_fraction)
