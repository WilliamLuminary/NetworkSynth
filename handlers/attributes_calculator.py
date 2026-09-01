# SPDX-License-Identifier: GPL-3.0-or-later
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import euclidean

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttributesCalculator:
    degree_distribution: Dict[int, float] = field(default_factory=dict)
    degree_transition_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    degree_lengths: Dict[int, List[float]] = field(default_factory=dict)
    degree_angles: Dict[int, List[float]] = field(default_factory=dict)
    average_length: float = 0.0
    average_degree: float = 0.0

    def analyze(self, graph: SynthGraph) -> "AttributesCalculator":
        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty. Skipping analysis.")
            return self

        precomputed_data = self._precompute_graph_data(graph)

        self.degree_distribution = self._compute_degree_distribution(
            precomputed_data["degrees"], graph.number_of_nodes()
        )
        self.degree_transition_probs = self._compute_degree_transition_probs(
            precomputed_data["degree_neighbors"]
        )
        self.degree_lengths = dict(precomputed_data["degree_to_lengths"])
        self.degree_angles = self._compute_degree_angles(
            precomputed_data["node_angles"], precomputed_data["degrees"]
        )
        self.average_length = self._compute_average_edge_length(
            precomputed_data["degree_to_lengths"]
        )
        self.average_degree = self._compute_average_degree(
            graph.number_of_edges(), graph.number_of_nodes()
        )
        return self

    @staticmethod
    def _precompute_graph_data(graph: SynthGraph):
        degrees: Dict[int, int] = {u: graph.degree(u) for u in graph.nodes()}
        positions = graph.positions()
        degree_neighbors = defaultdict(list)
        degree_to_lengths = defaultdict(list)
        node_angles = defaultdict(list)

        for node_u, node_v in graph.edges():
            degree_u = degrees[node_u]
            degree_v = degrees[node_v]
            degree_neighbors[degree_u].append(degree_v)
            degree_neighbors[degree_v].append(degree_u)

            u_pos = positions[node_u]
            v_pos = positions[node_v]
            length = euclidean(u_pos, v_pos)
            degree_to_lengths[degree_u].append(length)
            degree_to_lengths[degree_v].append(length)
            angle_u_to_v = (
                np.arctan2(v_pos[1] - u_pos[1], v_pos[0] - u_pos[0]) * 180 / np.pi
            )
            angle_v_to_u = (
                np.arctan2(u_pos[1] - v_pos[1], u_pos[0] - v_pos[0]) * 180 / np.pi
            )
            node_angles[node_u].append(angle_u_to_v)
            node_angles[node_v].append(angle_v_to_u)

        return {
            "degrees": degrees,
            "degree_neighbors": degree_neighbors,
            "degree_to_lengths": degree_to_lengths,
            "node_angles": node_angles,
        }

    @staticmethod
    def _compute_degree_distribution(
        degrees: Dict[int, int], num_nodes: int
    ) -> Dict[int, float]:
        return {
            deg: count / num_nodes for deg, count in Counter(degrees.values()).items()
        }

    @staticmethod
    def _compute_degree_transition_probs(
        degree_neighbors: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, float]]:
        return {
            degree: (
                {
                    neighbor_degree: count / len(neighbors_list)
                    for neighbor_degree, count in Counter(neighbors_list).items()
                }
                if neighbors_list
                else {}
            )
            for degree, neighbors_list in degree_neighbors.items()
        }

    @staticmethod
    def _compute_average_degree(num_edges: int, num_nodes: int) -> float:
        return 2.0 * num_edges / num_nodes

    @staticmethod
    def _compute_degree_angles(
        node_angles: Dict[int, List[float]], degrees: Dict[int, int]
    ) -> Dict[int, List[float]]:
        degree_angle_diffs = defaultdict(list)
        for node, angle_list in node_angles.items():
            if len(angle_list) < 2:
                continue
            sorted_angles = np.sort(angle_list)
            diffs = np.diff(sorted_angles)
            wrap_diff = 360.0 + sorted_angles[0] - sorted_angles[-1]
            diffs = np.append(diffs, wrap_diff)
            node_degree = degrees[node]
            degree_angle_diffs[node_degree].extend(diffs)
        return dict(degree_angle_diffs)

    @staticmethod
    def _compute_average_edge_length(
        degree_to_lengths: Dict[int, List[float]],
    ) -> float:
        total_length = 0.0
        total_count = 0
        for length_list in degree_to_lengths.values():
            total_length += sum(length_list)
            total_count += len(length_list)
        return total_length / total_count if total_count > 0 else 0.0
