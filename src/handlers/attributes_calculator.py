# src/handlers/attributes_calculator.py
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OldAttributesCalculator:
    degree_distribution: Dict[int, float] = field(default_factory=dict)
    degree_transition_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    degree_edge_lengths: Dict[int, List[float]] = field(default_factory=dict)
    degree_angle_diffs: Dict[int, List[float]] = field(default_factory=dict)
    average_edge_length: float = 0.0
    average_degree: float = 0.0

    def analyze(self, graph: SynthGraph) -> "OldAttributesCalculator":
        if graph.number_of_nodes() == 0:
            return self

        self.degree_distribution = dict(self._compute_degree_distribution(graph))
        self.degree_transition_probs = dict(
            self._compute_degree_transition_probs(graph)
        )
        self.average_degree = self._compute_average_degree(graph)
        (degree_edge_lengths, degree_angle_diffs, self.average_edge_length) = (
            self._compute_edge_lengths_and_angle_diffs(graph)
        )
        self.degree_edge_lengths = dict(degree_edge_lengths)
        self.degree_angle_diffs = dict(degree_angle_diffs)
        return self

    @staticmethod
    def _compute_degree_distribution(graph: SynthGraph) -> Dict[int, float]:
        degrees = [graph.degree(u) for u in graph.nodes()]
        total_nodes = graph.number_of_nodes()
        degree_counts = Counter(degrees)
        if total_nodes == 0:
            return {}
        return {
            int(degree): count / total_nodes for degree, count in degree_counts.items()
        }

    @staticmethod
    def _compute_degree_transition_probs(
        graph: SynthGraph,
    ) -> Dict[int, Dict[int, float]]:
        degree_neighbors = defaultdict(list)
        for node in graph.nodes():
            node_degree = graph.degree(node)
            neighbor_degrees = [graph.degree(nbr) for nbr in graph.neighbors(node)]
            degree_neighbors[node_degree].extend(neighbor_degrees)

        transition_probs = {}
        for degree, neighbor_degrees in degree_neighbors.items():
            counts = Counter(neighbor_degrees)
            total = sum(counts.values())
            if total == 0:
                transition_probs[degree] = {}
            else:
                transition_probs[degree] = {
                    neighbor_degree: count / total
                    for neighbor_degree, count in counts.items()
                }
        return transition_probs

    @staticmethod
    def _compute_edge_lengths_and_angle_diffs(
        graph: SynthGraph,
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], float]:
        positions = graph.positions()
        degree_to_lengths = defaultdict(list)
        degree_to_angle_diffs = defaultdict(list)

        total_length = 0.0
        total_edges_count = 0
        for node in graph.nodes():
            nbrs = graph.neighbors(node)
            if not nbrs:
                continue

            node_pos = positions[node]
            lengths = []
            angles = []

            for neighbor in nbrs:
                neighbor_pos = positions[neighbor]
                length = euclidean(node_pos, neighbor_pos)
                lengths.append(length)
                angle = (
                    np.arctan2(
                        neighbor_pos[1] - node_pos[1], neighbor_pos[0] - node_pos[0]
                    )
                    * 180
                    / np.pi
                )
                angles.append(angle)

            node_degree = graph.degree(node)
            degree_to_lengths[node_degree].extend(lengths)
            total_length += sum(lengths)
            total_edges_count += len(lengths)

            if len(angles) > 1:
                sorted_angles = np.sort(angles)
                angles_diff = np.diff(sorted_angles)
                angles_diff = np.append(
                    angles_diff, 360.0 + sorted_angles[0] - sorted_angles[-1]
                )
                degree_to_angle_diffs[node_degree].extend(angles_diff)

        avg_length = total_length / total_edges_count if total_edges_count > 0 else 0.0
        return degree_to_lengths, degree_to_angle_diffs, avg_length

    @staticmethod
    def _compute_average_degree(graph: SynthGraph):
        if graph.number_of_nodes() == 0:
            return 0.0
        degrees = [graph.degree(u) for u in graph.nodes()]
        return float(np.mean(degrees))


@dataclass(slots=True)
class AttributesCalculator:
    degree_distribution: Dict[int, float] = field(default_factory=dict)
    degree_transition_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    degree_lengths: Dict[int, List[float]] = field(default_factory=dict)
    degree_angles: Dict[int, List[float]] = field(default_factory=dict)
    average_length: float = 0.0
    average_degree: float = 0.0

    def analyze(self, graph: SynthGraph) -> "AttributesCalculator":
        """
        Analyze the graph to compute degree distribution, transition probabilities,
        edge lengths, angle differences, average edge length, and average degree.
        """
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
        """
        Count how many nodes have each degree, then normalize by total node count.
        """
        return {
            deg: count / num_nodes for deg, count in Counter(degrees.values()).items()
        }

    @staticmethod
    def _compute_degree_transition_probs(
        degree_neighbors: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, float]]:
        """
        For each degree d, gather the distribution of neighbor degrees
        and convert counts to probabilities.
        """
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
        """
        For undirected graphs, average_degree = (2 * E) / N.
        """
        return 2.0 * num_edges / num_nodes

    @staticmethod
    def _compute_degree_angles(
        node_angles: Dict[int, List[float]], degrees: Dict[int, int]
    ) -> Dict[int, List[float]]:
        """
        For each node, compute the angular differences between sorted angles
        and map them to that node's degree bucket.
        """
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
        """
        Compute the mean edge length from the 'degree_to_lengths' dictionary.
        Note: each edge length is counted twice (once from each node endpoint).
        """
        total_length = 0.0
        total_count = 0
        for length_list in degree_to_lengths.values():
            total_length += sum(length_list)
            total_count += len(length_list)
        return total_length / total_count if total_count > 0 else 0.0
