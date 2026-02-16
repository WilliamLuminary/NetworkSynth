import os
import pickle

import networkx as nx
import pytest

BASE_DIR = os.path.dirname(__file__)


@pytest.fixture
def load_weighted_test_nx_graph():
    file_path = os.path.realpath(
        os.path.join(BASE_DIR, "data", "A_10kX_weighted_network.pkl")
    )
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


@pytest.fixture
def load_unweighted_test_nx_graph():
    file_path = os.path.realpath(
        os.path.join(BASE_DIR, "data", "sample1_unweighted_network.pkl")
    )
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G
