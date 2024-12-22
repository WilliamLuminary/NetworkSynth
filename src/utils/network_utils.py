# src/properties/network_utils.py
import logging
import random
from collections import defaultdict
from typing import Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from config.base import BaseConfig

logger = logging.getLogger(__name__)


def _edge_lengths_and_weights(graph: nx.Graph) -> tuple:
    edge_lengths = []
    edge_weights = []

    for u, v, data in graph.edges(data=True):
        if 'length' not in data:
            pos_u = np.array(graph.nodes[u]['pos'], dtype=np.float64)
            pos_v = np.array(graph.nodes[v]['pos'], dtype=np.float64)
            length = np.linalg.norm(pos_u - pos_v)
            graph.edges[u, v]['length'] = length
        else:
            length = data['length']

        weight = data.get('weight', 1)

        edge_lengths.append(length)
        edge_weights.append(weight)

    return np.array(edge_lengths), np.array(edge_weights)


def _bins_and_baskets(edge_lengths, edge_weights, num_bins=100, method='thirds'):
    sorted_lengths = np.sort(edge_lengths)
    num_edges = len(sorted_lengths)

    if method == 'thirds':
        one_third = num_edges // 3
        two_thirds = 2 * num_edges // 3

        min_third_value = sorted_lengths[one_third]
        max_third_value = sorted_lengths[two_thirds]

        small_third_bins = np.linspace(sorted_lengths.min(), min_third_value, 50 + 1)
        middle_third_bins = np.linspace(min_third_value, max_third_value, 30 + 1)
        large_third_bins = np.linspace(max_third_value, sorted_lengths.max(), 20 + 1)

        length_bins = np.concatenate([small_third_bins, middle_third_bins[1:], large_third_bins[1:]])

    else:  # Default to 'linear' binning
        length_bins = np.linspace(edge_lengths.min(), edge_lengths.max(), num_bins + 1)

    weight_baskets = defaultdict(list)

    for length, weight in zip(edge_lengths, edge_weights):
        bin_idx = np.digitize(length, length_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(length_bins) - 2)
        weight_baskets[bin_idx].append(weight)

    return length_bins, weight_baskets


def _map_weight_by_length(length, length_bins, weight_baskets, y):
    bin_idx = np.digitize(length, length_bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(length_bins) - 2)
    return random.choice(weight_baskets[bin_idx]) if weight_baskets[bin_idx] else np.mean(y)


def mapping(graph: nx.Graph, plot: bool = False) -> tuple:
    edge_lengths, edge_weights = _edge_lengths_and_weights(graph)

    num_bins = 100
    length_bins, weight_baskets = _bins_and_baskets(edge_lengths, edge_weights, num_bins)
    mapped_weights = [_map_weight_by_length(length, length_bins, weight_baskets, edge_weights) for length in
                      edge_lengths]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(edge_lengths, edge_weights, c='blue', alpha=0.3, label='Original Data')
        plt.scatter(edge_lengths, mapped_weights, c='orange', alpha=0.3, label='Mapped Weights')
        plt.title('Edge Length vs Weight with Basket-Based Mapping')
        plt.xlabel('Length')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(False)
        plt.show()

    return mapped_weights, _map_weight_by_length, length_bins, weight_baskets, edge_weights


def calculate_frame(graph: nx.Graph = None, center_position: Union[tuple, list, ndarray] = None,
                    frame_range: int = BaseConfig.DEFAULT_FRAME_RANGE) -> list:
    if graph is None:
        if not center_position:
            raise ValueError("Either synthetic_graph or center_position must be provided.")
        _center_x, _center_y = center_position
    else:
        if center_position:
            raise ValueError("Only synthetic_graph or center_position must be provided.")
        _positions = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
        _center_x, _center_y = _positions[:, 0].mean(), _positions[:, 1].mean()

    _half_range = frame_range / 2
    frame = [
        [round(_center_x - _half_range, 2), round(_center_x + _half_range, 2)],
        [round(_center_y - _half_range, 2), round(_center_y + _half_range, 2)]
    ]
    return frame


def build_graph_pos_and_adj_mat(pos_and_adj_mat: tuple) -> nx.Graph:
    """
    :param pos_and_adj_mat: A tuple of positions and adjacency matrix.
    :return:
    """
    positions_of_nodes, adjacency_matrix = pos_and_adj_mat
    graph = nx.from_scipy_sparse_array(adjacency_matrix, edge_attribute='weight')
    for i, pos in enumerate(positions_of_nodes):
        graph.nodes[i]['pos'] = pos.astype(np.float64)

    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
    return graph


def build_graph_nodes_and_edges(nodes: Union[list, set], edges: Union[list, set]) -> nx.Graph:
    graph = nx.Graph()
    position_map = {node.position: node for node in nodes}
    for node in nodes:
        graph.add_node(position_map[node.position].id, pos=node.position)

    for edge in edges:
        u, v = position_map[edge[0]].id, position_map[edge[1]].id
        graph.add_edge(u, v)
    return graph
