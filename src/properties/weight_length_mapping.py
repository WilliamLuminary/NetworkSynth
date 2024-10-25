# properties/weight_length_mapping.py

import random
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt


def compute_edge_lengths_and_weights_from_graph(graph):
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


def create_bins_and_baskets(edge_lengths, edge_weights, num_bins=100, method='thirds'):
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


def map_length_to_weight(length, length_bins, weight_baskets, y):
    bin_idx = np.digitize(length, length_bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(length_bins) - 2)
    return random.choice(weight_baskets[bin_idx]) if weight_baskets[bin_idx] else np.mean(y)


def mapping(graph, plot=False):
    edge_lengths, edge_weights = compute_edge_lengths_and_weights_from_graph(graph)

    num_bins = 100
    length_bins, weight_baskets = create_bins_and_baskets(edge_lengths, edge_weights, num_bins)
    mapped_weights = [map_length_to_weight(length, length_bins, weight_baskets, edge_weights) for length in
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

    return mapped_weights, map_length_to_weight, length_bins, weight_baskets, edge_weights
