# src/handlers/attributes_calculator.py
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AttributesCalculator:
    degree_distribution: Dict[int, float] = field(default_factory=dict)
    degree_transition_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    degree_edge_lengths: Dict[int, List[float]] = field(default_factory=dict)
    degree_angle_diffs: Dict[int, List[float]] = field(default_factory=dict)
    average_edge_length: float = 0.0
    average_degree: float = 0.0

    def analyze(self, graph: nx.Graph) -> "AttributesCalculator":
        if graph.number_of_nodes() == 0:
            return self

        self.degree_distribution = dict(self._compute_degree_distribution(graph))
        self.degree_transition_probs = dict(self._compute_degree_transition_probs(graph))
        self.average_degree = self._compute_average_degree(graph)
        (degree_edge_lengths,
         degree_angle_diffs,
         self.average_edge_length) = self._compute_edge_lengths_and_angle_diffs(graph)
        self.degree_edge_lengths = dict(degree_edge_lengths)
        self.degree_angle_diffs = dict(degree_angle_diffs)
        return self

    @staticmethod
    def _compute_degree_distribution(graph: nx.Graph) -> Dict[int, float]:
        degrees = [deg for _, deg in graph.degree()]
        total_nodes = graph.number_of_nodes()
        degree_counts = Counter(degrees)
        if total_nodes == 0:
            return {}
        return {int(degree): count / total_nodes for degree, count in degree_counts.items()}

    @staticmethod
    def _compute_degree_transition_probs(graph: nx.Graph) -> Dict[int, Dict[int, float]]:
        degrees = dict(graph.degree())
        degree_neighbors = defaultdict(list)
        for node, adjacency_dict in graph.adjacency():
            node_degree = degrees[node]
            neighbor_degrees = [degrees[nbr] for nbr in adjacency_dict.keys()]
            degree_neighbors[node_degree].extend(neighbor_degrees)

        transition_probs = {}
        for degree, neighbor_degrees in degree_neighbors.items():
            counts = Counter(neighbor_degrees)
            total = sum(counts.values())
            if total == 0:
                transition_probs[degree] = {}
            else:
                transition_probs[degree] = {
                    neighbor_degree: count / total for neighbor_degree, count in counts.items()
                }
        return transition_probs

    @staticmethod
    def _compute_edge_lengths_and_angle_diffs(
            graph: nx.Graph
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], float]:
        node_positions = nx.get_node_attributes(graph, 'pos')
        degree_to_lengths = defaultdict(list)
        degree_to_angle_diffs = defaultdict(list)

        total_length = 0.0
        total_edges_count = 0
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue

            node_pos = np.array(node_positions[node], dtype=np.float64)

            lengths = []
            angles = []

            for neighbor in neighbors:
                neighbor_pos = np.array(node_positions[neighbor], dtype=np.float64)

                length = euclidean(node_pos, neighbor_pos)
                lengths.append(length)

                angle = np.arctan2(neighbor_pos[1] - node_pos[1], neighbor_pos[0] - node_pos[0]) * 180 / np.pi
                angles.append(angle)

            node_degree = graph.degree[node]
            degree_to_lengths[node_degree].extend(lengths)

            total_length += sum(lengths)
            total_edges_count += len(lengths)

            if len(angles) > 1:
                sorted_angles = np.sort(angles)
                angles_diff = np.diff(sorted_angles)
                angles_diff = np.append(angles_diff, 360.0 + sorted_angles[0] - sorted_angles[-1])
                degree_to_angle_diffs[node_degree].extend(angles_diff)

        avg_length = total_length / total_edges_count if total_edges_count > 0 else 0.0
        return degree_to_lengths, degree_to_angle_diffs, avg_length

    @staticmethod
    def _compute_average_degree(graph: nx.Graph):
        if graph.number_of_nodes() == 0:
            return 0.0
        degrees = [deg for _, deg in graph.degree()]
        return float(np.mean(degrees))


@dataclass(slots=True)
class NewAttributesCalculator:
    degree_distribution: Dict[int, float] = field(default_factory=dict)
    degree_transition_probs: Dict[int, Dict[int, float]] = field(default_factory=dict)
    degree_edge_lengths: Dict[int, List[float]] = field(default_factory=dict)
    degree_angle_diffs: Dict[int, List[float]] = field(default_factory=dict)
    average_edge_length: float = 0.0
    average_degree: float = 0.0

    def analyze(self, graph: nx.Graph):
        if graph.number_of_nodes() == 0:
            return self

        precomputed = self._precompute_graph_data(graph)

        self.degree_distribution = dict(self._compute_degree_distribution(
            precomputed["degrees"], precomputed["n_nodes"]
        ))
        self.degree_transition_probs = dict(self._compute_degree_transition_probs(
            precomputed["degree_neighbors"]
        ))
        self.average_degree = self._compute_average_degree(
            graph.number_of_edges(), precomputed["n_nodes"]
        )
        (
            degree_edge_lengths,
            degree_angle_diffs,
            self.average_edge_length
        ) = self._compute_edge_lengths_and_angle_diffs(
            precomputed["degree_to_lengths"],
            precomputed["node_angles"],
            precomputed["degrees"]
        )
        self.degree_edge_lengths = dict(degree_edge_lengths)
        self.degree_angle_diffs = dict(degree_angle_diffs)
        return self

    @staticmethod
    def _precompute_graph_data(graph: nx.Graph):
        degrees = dict(graph.degree())
        n_nodes = graph.number_of_nodes()

        pos = nx.get_node_attributes(graph, "pos")
        degree_neighbors = defaultdict(list)
        degree_to_lengths = defaultdict(list)
        node_angles = defaultdict(list)

        for u, v in graph.edges():
            du = degrees[u]
            dv = degrees[v]
            degree_neighbors[du].append(dv)
            degree_neighbors[dv].append(du)

            if u in pos and v in pos:
                u_pos = np.array(pos[u], dtype=np.float64)
                v_pos = np.array(pos[v], dtype=np.float64)

                length = euclidean(u_pos, v_pos)
                degree_to_lengths[du].append(length)
                degree_to_lengths[dv].append(length)

                angle_uv = np.arctan2(v_pos[1] - u_pos[1], v_pos[0] - u_pos[0]) * 180 / np.pi
                angle_vu = np.arctan2(u_pos[1] - v_pos[1], u_pos[0] - v_pos[0]) * 180 / np.pi
                node_angles[u].append(angle_uv)
                node_angles[v].append(angle_vu)

        return {
            "degrees": degrees,
            "n_nodes": n_nodes,
            "degree_neighbors": degree_neighbors,
            "degree_to_lengths": degree_to_lengths,
            "node_angles": node_angles,
        }

    @staticmethod
    def _compute_degree_distribution(degrees: Dict, n_nodes: int) -> Dict[int, float]:
        deg_counts = Counter(degrees.values())
        return {d: cnt / n_nodes for d, cnt in deg_counts.items()} if n_nodes else {}

    @staticmethod
    def _compute_degree_transition_probs(
            degree_neighbors: Dict[int, List[int]]
    ) -> Dict[int, Dict[int, float]]:

        transition_probs = {}
        for deg, neighbors_list in degree_neighbors.items():
            total = len(neighbors_list)
            if total == 0:
                transition_probs[deg] = {}
            else:
                deg_count = Counter(neighbors_list)
                transition_probs[deg] = {
                    nd: c / total for nd, c in deg_count.items()
                }
        return transition_probs

    @staticmethod
    def _compute_average_degree(n_edges: int, n_nodes: int) -> float:
        return 2.0 * n_edges / n_nodes if n_nodes else 0.0

    @staticmethod
    def _compute_edge_lengths_and_angle_diffs(
            degree_to_lengths: Dict[int, List[float]],
            node_angles: Dict[int, List[float]],
            degrees: Dict
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], float]:
        total_length = 0.0
        total_count = 0
        for deg, length_list in degree_to_lengths.items():
            total_length += sum(length_list)
            total_count += len(length_list)
        avg_length = (total_length / total_count) if total_count else 0.0

        degree_angle_diffs = defaultdict(list)
        for node, angles in node_angles.items():
            if len(angles) < 2:
                continue
            sorted_angles = np.sort(angles)
            diffs = np.diff(sorted_angles)
            diffs = np.append(diffs, 360 + sorted_angles[0] - sorted_angles[-1])
            deg = degrees[node]
            degree_angle_diffs[deg].extend(diffs)

        return (
            dict(degree_to_lengths),
            dict(degree_angle_diffs),
            avg_length,
        )
