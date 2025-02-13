import logging
import random
from collections import defaultdict
from typing import DefaultDict, Final, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

NUM_BINS: Final[int] = 100


class Mapper:
    def __init__(self, graph: nx.Graph):
        lengths, weights = self._compute_edge_metrics(graph)
        self.avg_weights: Final[float] = np.mean(weights)
        self.length_bins, self.weight_distributions = self._create_map_metrics(lengths, weights)

    def _length_to_weight(self, length: float) -> float:
        bin_idx = np.digitize(length, self.length_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.length_bins) - 2)
        distribution = self.weight_distributions.get(bin_idx, None)
        return random.choice(distribution) if distribution else self.avg_weights

    @staticmethod
    def _compute_edge_metrics(graph: nx.Graph) -> tuple:
        lengths, weights = [], []
        for u, v, data in graph.edges(data=True):
            if 'length' not in data:
                length = euclidean(graph.nodes[u]['pos'], graph.nodes[v]['pos'])
                data['length'] = length
            np.append(lengths, data['length'])
            np.append(weights, data.get('weight', 1.0))
        return lengths, weights

    @staticmethod
    def _create_map_metrics(lengths: np.ndarray,
                            weights: np.ndarray
                            ) -> Tuple[np.ndarray, DefaultDict[int, list]]:

        sorted_lengths = np.sort(lengths)
        length_bins = Mapper._create_thirds_bins(sorted_lengths) \
            if len(sorted_lengths) > 100 \
            else np.linspace(sorted_lengths[0], sorted_lengths[1], NUM_BINS + 1)

        weight_distributions = defaultdict(list)
        for length, weight in zip(lengths, weights):
            bin_idx = np.digitize(length, length_bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(length_bins) - 2)
            weight_distributions[bin_idx].append(weight)
        return length_bins, weight_distributions

    @staticmethod
    def _create_thirds_bins(sorted_lengths):
        n = len(sorted_lengths)
        lower, upper = sorted_lengths[n // 3], sorted_lengths[2 * n // 3]
        length_bins = np.concatenate([
            np.linspace(sorted_lengths[0], lower, 51),
            np.linspace(lower, upper, 31)[1:],
            np.linspace(upper, sorted_lengths[-1], 21)[1:]
        ])
        return length_bins

    def assign_weights(self, graph: nx.Graph):
        if nx.get_edge_attributes(graph, 'weight'):
            raise ValueError('Graph already has edge weights assigned')

        lengths = []
        for u, v, data in graph.edges(data=True):
            if 'length' not in data:
                length = euclidean(graph.nodes[u]['pos'], graph.nodes[v]['pos'])
                data['length'] = length
            lengths.append(data['length'])
        weights = [self._length_to_weight(length) for length in lengths]
        nx.set_edge_attributes(graph, dict(zip(graph.edges(), weights)), 'weight')
