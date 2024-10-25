# graph_properties.py

from collections import defaultdict, Counter

import numpy as np
from scipy.spatial.distance import euclidean


class GraphProperties:
    def __init__(self, graph):
        self.graph = graph
        self.degree_distribution = None
        self.degree_transition_probs = None
        self.degree_angles = None
        self.degree_edge_lengths = None
        self.avg_degree = None
        self.avg_length = None

    def compute_properties(self):
        self.compute_degree_distribution()
        self.compute_degree_transition_probs()
        self.compute_edge_lengths_and_angles()
        self.compute_average_degree()

    def compute_degree_distribution(self):
        degrees = [d for n, d in self.graph.degree()]
        degree_counts = Counter(degrees)
        total_nodes = self.graph.number_of_nodes()
        self.degree_distribution = {k: v / total_nodes for k, v in degree_counts.items()}

    def compute_degree_transition_probs(self):
        degree_neighbors = defaultdict(list)
        degrees = dict(self.graph.degree())
        for node, neighbors in self.graph.adjacency():
            node_degree = degrees[node]
            neighbor_degrees = [degrees[neighbor] for neighbor in neighbors]
            degree_neighbors[node_degree].extend(neighbor_degrees)

        self.degree_transition_probs = {}
        for degree, neighbor_degrees in degree_neighbors.items():
            counts = Counter(neighbor_degrees)
            total = sum(counts.values())
            self.degree_transition_probs[degree] = {k: v / total for k, v in counts.items()}

    def compute_edge_lengths_and_angles(self):
        self.degree_edge_lengths = defaultdict(list)
        self.degree_angles = defaultdict(list)
        total_length = 0
        count = 0
        positions = {n: np.array(self.graph.nodes[n]['pos'], dtype=np.float64) for n in self.graph.nodes()}

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue

            node_pos = positions[node]
            lengths = []
            angles = []
            for neighbor in neighbors:
                neighbor_pos = positions[neighbor]
                length = euclidean(node_pos, neighbor_pos)
                lengths.append(length)
                angle = np.arctan2(neighbor_pos[1] - node_pos[1], neighbor_pos[0] - node_pos[0]) * 180 / np.pi
                angles.append(angle)

            degree = self.graph.degree[node]
            self.degree_edge_lengths[degree].extend(lengths)
            total_length += sum(lengths)
            count += len(lengths)

            if len(angles) > 1:
                sorted_angles = np.sort(angles)
                angles_diff = np.diff(sorted_angles)
                angles_diff = np.append(angles_diff, 360 + sorted_angles[0] - sorted_angles[-1])
                self.degree_angles[degree].extend(angles_diff)

        self.avg_length = total_length / count if count > 0 else 0

    def compute_average_degree(self):
        degrees = [d for n, d in self.graph.degree()]
        self.avg_degree = np.mean(degrees)
