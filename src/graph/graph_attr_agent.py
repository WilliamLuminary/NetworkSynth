# src/original_network/graph_attr_agent.py

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


class GraphAttrAgent:
    def __init__(self):
        self.node_positions: Dict[int, Tuple[float, float]] = {}
        self.degree_distribution: Dict[int, float] = {}
        self.degree_transition_probs: Dict[int, Dict[int, float]] = {}
        self.degree_edge_lengths: Dict[int, List[float]] = {}
        self.degree_angle_diffs: Dict[int, List[float]] = {}
        self.average_edge_length: float = 0.0
        self.average_degree: float = 0.0

    def analyze(self, graph: nx.Graph):
        self.node_positions = nx.get_node_attributes(graph, 'pos')
        self.degree_distribution = self._compute_degree_distribution(graph)
        self.degree_transition_probs = self._compute_degree_transition_probs(graph)
        (self.degree_edge_lengths,
         self.degree_angle_diffs,
         self.average_edge_length) = self._compute_edge_lengths_and_angle_diffs(graph)
        self.average_degree = self._compute_average_degree(graph)

    def _to_dict(self) -> dict:
        return {
            'node_positions': self.node_positions,
            'degree_distribution': self.degree_distribution,
            'degree_transition_probs': self.degree_transition_probs,
            'degree_edge_lengths': self.degree_edge_lengths,
            'degree_angle_diffs': self.degree_angle_diffs,
            'average_edge_length': self.average_edge_length,
            'average_degree': self.average_degree
        }

    def as_savable(self):
        return self._to_dict()

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

    def _compute_edge_lengths_and_angle_diffs(
            self, graph: nx.Graph
    ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], float]:
        degree_to_lengths = defaultdict(list)
        degree_to_angle_diffs = defaultdict(list)

        total_length = 0.0
        total_edges_count = 0

        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue

            node_pos = np.array(self.node_positions[node], dtype=np.float64)

            lengths = []
            angles = []

            for neighbor in neighbors:
                neighbor_pos = np.array(self.node_positions[neighbor], dtype=np.float64)

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
