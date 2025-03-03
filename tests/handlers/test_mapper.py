import os
import pickle
import time
from time import sleep

import networkx as nx
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from config import GenConfig1
from handlers import EnhancedMapper, Mapper

GenConfig1.initialize()


@pytest.fixture
def load_graph_from_pickle():
    file_path = os.path.realpath(os.path.join("..", "data", "A_10kX_weighted_network.pkl"))
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


def profile_mapper_time(mapper, graph, iterations=10):
    """
    :return Mean time, Standard deviation.
    """
    times = []
    for _ in range(iterations):
        graph_copy = graph.copy()
        # Remove weights for a fair test.
        for _, _, data in graph_copy.edges(data=True):
            data.pop('weight', None)
        start = time.perf_counter()
        mapper.assign_weights(graph_copy)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def compute_quality(mapper, graph):
    """
    :return: Mean Absolute Error (MAE), Pearson correlation.
    """
    # noinspection PyProtectedMember
    original_lengths, original_weights = Mapper._compute_edge_metrics(graph)
    graph_copy = graph.copy()
    for _, _, data in graph_copy.edges(data=True):
        data.pop('weight', None)
    mapper.assign_weights(graph_copy)

    # noinspection PyProtectedMember
    new_lengths, new_weights = Mapper._compute_edge_metrics(graph_copy)
    mae = np.mean(np.abs(np.array(new_weights) - np.array(original_weights)))
    corr, _ = pearsonr(original_weights, new_weights)
    return mae, corr


def test_mapper(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    mapper = Mapper(sample_graph)
    plot_map(mapper, sample_graph)


def plot_map(mapper, sample_graph):
    # noinspection PyProtectedMember
    ori_lengths, ori_weights = Mapper._compute_edge_metrics(sample_graph)
    new_graph = sample_graph.copy()
    for _ in range(10):
        for u, v, data in new_graph.edges(data=True):
            del data['weight']
        mapper.assign_weights(new_graph)
        # noinspection PyProtectedMember
        new_lengths, new_weights = Mapper._compute_edge_metrics(new_graph)
        plt.figure(figsize=(8, 6))
        plt.scatter(ori_lengths, ori_weights, c='blue', alpha=0.3, label='Original Data')
        plt.scatter(new_lengths, new_weights, c='orange', alpha=0.3, label='Mapped Weights')
        plt.title('Edge Length vs Weight with Basket-Based Mapping')
        plt.xlabel('Length')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(False)
        plt.show()
        print("Check the plot for the comparison of original and mapped weights.")
        plt.close()
        sleep(0.1)


def test_enhanced_mapper(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    mapper = EnhancedMapper(sample_graph)
    plot_map(mapper, sample_graph)


def test_mapper_performance(load_graph_from_pickle):
    """Measures runtime."""
    sample_graph = load_graph_from_pickle
    mapper1 = Mapper(sample_graph)
    mapper2 = EnhancedMapper(sample_graph)

    time1, std1 = profile_mapper_time(mapper1, sample_graph)
    time2, std2 = profile_mapper_time(mapper2, sample_graph)

    print()
    print(f"Mapper: Avg Time = {time1:.4f}s ± {std1:.4f}s")
    print(f"Enhanced Mapper: Avg Time = {time2:.4f}s ± {std2:.4f}s")


def test_mapper_quality(load_graph_from_pickle):
    """Measures Mean absolute error and Pearson correlation."""
    sample_graph = load_graph_from_pickle
    mapper1 = Mapper(sample_graph)
    mapper2 = EnhancedMapper(sample_graph)

    mae1, corr1 = compute_quality(mapper1, sample_graph)
    mae2, corr2 = compute_quality(mapper2, sample_graph)

    print()
    print(f"Mapper: MAE = {mae1:.4f}, Corr = {corr1:.4f}")
    print(f"Enhanced Mapper: MAE = {mae2:.4f}, Corr = {corr2:.4f}")
    # assert abs(corr1 - corr2) < 0.1
