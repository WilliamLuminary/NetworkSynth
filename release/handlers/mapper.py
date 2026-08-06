import logging
import random
from collections import defaultdict
from typing import DefaultDict, Final, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)

NUM_BINS: Final[int] = 100


class Mapper:
    def __init__(self, graph: SynthGraph):
        if graph.is_weighted():
            lengths, weights = self._compute_edge_metrics(graph)
            self.avg_weights: Final = np.mean(weights)
            bins, dist = self._create_mapping_metrics(lengths, weights)
            self.length_bins: Final = bins
            self.weight_distributions: Final = dist
            self.original_lengths = np.array(lengths)
            self.original_weights = np.array(weights)
            self.skipped = False
        else:
            self.skipped = True
            logger.info("Input graph has no weights.")

    @staticmethod
    def _compute_edge_metrics(graph: SynthGraph):
        positions = graph.positions()
        lengths, weights = [], []
        for u, v, w in graph.edges_with_weights():
            length = euclidean(positions[u], positions[v])
            lengths.append(length)
            weights.append(w)
        return lengths, weights

    @staticmethod
    def _create_mapping_metrics(
        lengths, weights
    ) -> Tuple[np.ndarray, DefaultDict[int, list]]:
        sorted_lengths = np.sort(lengths)
        length_bins = (
            Mapper._create_thirds_bins(sorted_lengths)
            if len(sorted_lengths) > 100
            else Mapper._create_linear_bins(sorted_lengths)
        )

        return Mapper._build_weight_distributions(length_bins, lengths, weights)

    @staticmethod
    def _build_weight_distributions(length_bins, lengths, weights):
        weight_distributions = defaultdict(list)
        bin_indices = np.clip(
            np.digitize(lengths, length_bins) - 1, 0, len(length_bins) - 2
        )
        for idx, weight in zip(bin_indices, weights):
            weight_distributions[idx].append(weight)
        return length_bins, weight_distributions

    @staticmethod
    def _create_linear_bins(sorted_lengths):
        return np.linspace(sorted_lengths[0], sorted_lengths[-1], NUM_BINS + 1)

    @staticmethod
    def _create_thirds_bins(sorted_lengths):
        n = len(sorted_lengths)
        lower, upper = sorted_lengths[n // 3], sorted_lengths[2 * n // 3]
        length_bins = np.concatenate(
            [
                np.linspace(sorted_lengths[0], lower, 51),
                np.linspace(lower, upper, 31)[1:],
                np.linspace(upper, sorted_lengths[-1], 21)[1:],
            ]
        )
        return length_bins

    def assign_weights(self, graph: SynthGraph) -> None:
        """
        Assigns edge weights to the graph based on edge lengths.
        Converts the graph to weighted in-place if needed.
        """
        if self.skipped:
            return

        if graph.is_weighted():
            raise ValueError("Graph already has edge weights assigned")

        graph.make_weighted()
        positions = graph.positions()
        for u, v in graph.edges():
            length = euclidean(positions[u], positions[v])
            graph.set_weight(u, v, self._length_to_weight(length))

    def _length_to_weight(self, length: float) -> float:
        bin_idx = np.clip(
            np.digitize(length, self.length_bins) - 1, 0, len(self.length_bins) - 2
        )
        distribution = self.weight_distributions.get(bin_idx)
        return random.choice(distribution) if distribution else self.avg_weights


class EnhancedMapper(Mapper):
    def __init__(self, graph: SynthGraph, *, mix_intensity: float = 0.3):
        """
        :param mix_intensity: 0.0 (original) to 1.0 (full mixing)
        """
        super().__init__(graph)
        self.mix_intensity = np.clip(mix_intensity, 0, 1)
        self._prepare_mixing_model()

    def _prepare_mixing_model(self):
        lengths = np.array(self.original_lengths)
        weights = np.array(self.original_weights)

        from scipy.stats import gaussian_kde

        self.kde = gaussian_kde(np.vstack([lengths, weights]))
        from sklearn.neighbors import NearestNeighbors

        self.nn_model = NearestNeighbors(n_neighbors=50).fit(
            np.vstack([lengths, weights]).T
        )

        self.bandwidth = self.mix_intensity * np.std(lengths) * 2

    def _get_mixed_weight(self, length: float) -> float:
        distances, indices = self.nn_model.kneighbors(
            [[length, 0]], return_distance=True
        )

        raw_weights = 1 / (distances.squeeze() + 1e-8)
        mixing_probs = raw_weights / raw_weights.sum()

        return np.random.choice(
            self.original_weights[indices.squeeze()], p=mixing_probs
        )

    def assign_weights(self, graph: SynthGraph) -> None:
        if graph.is_weighted():
            raise ValueError("Graph already has edge weights assigned")

        graph.make_weighted()
        positions = graph.positions()
        for u, v in graph.edges():
            length = euclidean(positions[u], positions[v])
            if np.random.random() < self.mix_intensity:
                weight = self._get_mixed_weight(length)
            else:
                weight = self._length_to_weight(length)
            graph.set_weight(u, v, weight)
