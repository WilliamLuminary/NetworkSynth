# tests/handlers/test_mapper.py
import os
import pickle
from time import sleep

import networkx as nx
import pytest
from matplotlib import pyplot as plt

from config import Config1
from handlers import Mapper, EnhancedMapper

Config1.initialize()


@pytest.fixture
def load_graph_from_pickle():
    file_path = os.path.realpath(os.path.join("..", "data", "A_10kX_weighted_network.pkl"))
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


def test_mapper(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    mapper = Mapper(sample_graph)
    ori_lengths, ori_weights = Mapper._compute_edge_metrics(sample_graph)

    new_graph = sample_graph.copy()
    for _ in range(10):
        for u, v, data in new_graph.edges(data=True):
            del data['weight']
        mapper.assign_weights(new_graph)
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
        sleep(.1)


def test_enhanced_mapper(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    mapper = EnhancedMapper(sample_graph)
    ori_lengths, ori_weights = Mapper._compute_edge_metrics(sample_graph)

    new_graph = sample_graph.copy()
    for _ in range(10):
        for u, v, data in new_graph.edges(data=True):
            del data['weight']
        mapper.assign_weights(new_graph)
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
        sleep(.1)
