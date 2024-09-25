from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import euclidean


def group_nodes_by_degree(graph):
    degree_count = dict(graph.degree())
    nodes_by_degree = defaultdict(list)
    for node, degree in degree_count.items():
        nodes_by_degree[degree].append(node)
    return degree_count, nodes_by_degree


def calculate_degree_probabilities(nodes_by_degree, total_nodes):
    return {degree: len(nodes) / total_nodes for degree, nodes in nodes_by_degree.items()}


def calculate_degree_transition_probs(graph, degree_count):
    degree_neighbors_map = defaultdict(list)
    for node, neighbors in graph.adjacency():
        node_degree = degree_count[node]
        neighbor_degrees = [degree_count[neighbor] for neighbor in neighbors]
        degree_neighbors_map[node_degree].extend(neighbor_degrees)

    degree_transition_probs = {}
    for degree, neighbor_degrees in degree_neighbors_map.items():
        total_count = sum(Counter(neighbor_degrees).values())
        probs = {k: v / total_count for k, v in Counter(neighbor_degrees).items()}
        total_prob = sum(probs.values())
        assert np.isclose(total_prob,
                          1.0), f"Total probability for degree {degree} does not sum to 1 after normalization."
        degree_transition_probs[degree] = probs

    return degree_transition_probs


def compute_and_store_edge_lengths_and_angles(graph):
    total_length_sum = 0
    total_length_count = 0
    edge_lengths_by_degree = defaultdict(list)
    angles_by_degree = defaultdict(list)

    normalized_positions = {node: np.array(graph.nodes[node]['pos'], dtype=np.float64) for node in graph.nodes()}

    for node, node_pos in normalized_positions.items():
        neighbors = list(graph.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors == 0:
            continue  # Skip isolated nodes

        lengths = [euclidean(node_pos, normalized_positions[neighbor]) for neighbor in neighbors]
        node_degree = graph.degree[node]

        edge_lengths_by_degree[node_degree].extend(lengths)
        total_length_sum += np.sum(lengths)
        total_length_count += len(lengths)

        if num_neighbors > 1:
            polar_coords = [(neighbor, np.arctan2(normalized_positions[neighbor][1] - node_pos[1],
                                                  normalized_positions[neighbor][0] - node_pos[0]))
                            for neighbor in neighbors]
            polar_coords.sort(key=lambda x: x[1])

            angles = []
            for i in range(len(polar_coords)):
                angle_diff = (polar_coords[(i + 1) % len(polar_coords)][1] - polar_coords[i][1]) * (180 / np.pi)
                if angle_diff < 0:
                    angle_diff += 360
                angles.append(round(angle_diff, 0))
            angles_by_degree[node_degree].extend(angles)

    average_length = total_length_sum / total_length_count if total_length_count > 0 else 0
    return edge_lengths_by_degree, angles_by_degree, average_length


def calculate_degree_distribution(graph):
    degree_count, nodes_by_degree = group_nodes_by_degree(graph)
    total_nodes = sum(len(nodes) for nodes in nodes_by_degree.values())
    degree_distribution = calculate_degree_probabilities(nodes_by_degree, total_nodes)
    return degree_distribution


def calculate_average_degree(graph):
    degree_count = dict(graph.degree())
    avg_degree = np.mean(list(degree_count.values()))
    return avg_degree


def prepare_graph_properties(graph):
    degree_distribution = calculate_degree_distribution(graph)
    degree_count, _ = group_nodes_by_degree(graph)
    degree_transition_probs = calculate_degree_transition_probs(graph, degree_count)
    degree_edge_lengths, degree_angles, average_length = compute_and_store_edge_lengths_and_angles(graph)
    avg_degree = calculate_average_degree(graph)
    return degree_distribution, degree_transition_probs, degree_angles, degree_edge_lengths, avg_degree, average_length