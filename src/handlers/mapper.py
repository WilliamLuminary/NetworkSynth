import logging
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class MapHandler:
    mapped_weights = None
    length_bins = None
    weight_baskets = None
    edge_weights = None

    def __init__(self, graph: nx.Graph):
        self._initialize_mapper(graph)
        self.assign_weights(graph)

    def _map_weight_by_length(self, length):
        bin_idx = np.digitize(length, self.length_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.length_bins) - 2)
        return random.choice(self.weight_baskets[bin_idx]) if self.weight_baskets[bin_idx] else np.mean(
            self.edge_weights)

    def _initialize_mapper(self, graph: nx.Graph, plot: bool = False):
        edge_lengths, self.edge_weights = self._edge_lengths_and_weights(graph)

        num_bins = 100
        self.length_bins, self.weight_baskets = self._bins_and_baskets(edge_lengths, self.edge_weights, num_bins)

        if plot:
            mapped_weights = [self._map_weight_by_length(length)
                              for length in edge_lengths]
            plt.figure(figsize=(8, 6))
            plt.scatter(edge_lengths, self.edge_weights, c='blue', alpha=0.3, label='Original Data')
            plt.scatter(edge_lengths, mapped_weights, c='orange', alpha=0.3, label='Mapped Weights')
            plt.title('Edge Length vs Weight with Basket-Based Mapping')
            plt.xlabel('Length')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(False)
            plt.show()

    def assign_weights(self, graph: nx.Graph):
        edge_lengths = [euclidean(graph.nodes[u]['pos'], graph.nodes[v]['pos'])
                        for u, v in graph.edges()]
        weights = [self._map_weight_by_length(length) for length in edge_lengths]
        for (u, v), weight in zip(graph.edges(), weights):
            graph[u][v]['weight'] = weight

    @staticmethod
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

    @staticmethod
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
