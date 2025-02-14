import os
import pickle
from typing import Dict

import networkx as nx
import pytest

from graph import GraphAttrAgent


@pytest.fixture
def load_graph_from_pickle():
    file_path = os.path.realpath(os.path.join("..", "data", "A_10kX_weighted_network.pkl"))
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


def test_pickle_dump_graph_attr_agent(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    agent = GraphAttrAgent(sample_graph)
    agent.analyze()
    with open("attr_dict.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(agent, f)

    assert os.path.exists("attr_dict.pkl")
    with open("attr_dict.pkl", "rb") as f:
        load_file = pickle.load(f)
    assert isinstance(load_file, Dict)
