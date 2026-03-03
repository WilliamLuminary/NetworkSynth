import time
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from configs import GenConfig1
from handlers import Mapper

GenConfig1.initialize()


def profile_mapper_time(mapper, graph, iterations=10):
    """
    :return Mean time, Standard deviation.
    """
    times = []
    for _ in range(iterations):
        graph_copy = graph.copy()
        graph_copy.make_unweighted()
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
    graph_copy.make_unweighted()
    mapper.assign_weights(graph_copy)

    # noinspection PyProtectedMember
    new_lengths, new_weights = Mapper._compute_edge_metrics(graph_copy)
    mae = np.mean(np.abs(np.array(new_weights) - np.array(original_weights)))
    corr, _ = pearsonr(original_weights, new_weights)
    return mae, corr


def plot_map(mapper, sample_graph):
    # noinspection PyProtectedMember
    ori_lengths, ori_weights = Mapper._compute_edge_metrics(sample_graph)
    for _ in range(10):
        new_graph = sample_graph.copy()
        new_graph.make_unweighted()
        mapper.assign_weights(new_graph)
        # noinspection PyProtectedMember
        new_lengths, new_weights = Mapper._compute_edge_metrics(new_graph)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            ori_lengths, ori_weights, c="blue", alpha=0.3, label="Original Data"
        )
        plt.scatter(
            new_lengths, new_weights, c="orange", alpha=0.3, label="Mapped Weights"
        )
        plt.title("Edge Length vs Weight with Basket-Based Mapping")
        plt.xlabel("Length")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(False)
        plt.show()
        print("Check the plot for the comparison of original and mapped weights.")
        plt.close()
        sleep(0.1)


def test_mapper(load_weighted_test_synth_graph):
    sample_graph = load_weighted_test_synth_graph
    mapper = Mapper(sample_graph)
    plot_map(mapper, sample_graph)


def test_mapper_performance(load_weighted_test_synth_graph):
    """Measures runtime."""
    sample_graph = load_weighted_test_synth_graph
    mapper1 = Mapper(sample_graph)

    time1, std1 = profile_mapper_time(mapper1, sample_graph)

    print()
    print(f"Mapper: Avg Time = {time1:.4f}s ± {std1:.4f}s")


def test_mapper_quality(load_weighted_test_synth_graph):
    """Measures Mean absolute error and Pearson correlation."""
    sample_graph = load_weighted_test_synth_graph
    mapper1 = Mapper(sample_graph)

    mae1, corr1 = compute_quality(mapper1, sample_graph)

    print()
    print(f"Mapper: MAE = {mae1:.4f}, Corr = {corr1:.4f}")
    # assert abs(corr1 - corr2) < 0.1
