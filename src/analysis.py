from collections import defaultdict

import numpy as np


def create_bins_and_baskets(edge_lengths, edge_weights, num_bins=100):
    length_bins = np.linspace(edge_lengths.min(), edge_lengths.max(), num_bins + 1)
    weight_baskets = defaultdict(list)
    for length, weight in zip(edge_lengths, edge_weights):
        bin_idx = np.digitize(length, length_bins) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        weight_baskets[bin_idx].append(weight)
    return length_bins, weight_baskets


def mapping(G, plot=False):
    edge_lengths, edge_weights = compute_edge_lengths_and_weights_from_graph(G)
    length_bins, weight_baskets = create_bins_and_baskets(edge_lengths, edge_weights)
    mapped_weights = [map_length_to_weight(length, length_bins, weight_baskets, edge_weights) for length in
                      edge_lengths]
    # Plotting and other functionalities...
