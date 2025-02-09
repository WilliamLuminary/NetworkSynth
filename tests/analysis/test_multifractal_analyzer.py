import os
import pickle

import networkx as nx
import numpy as np
import pytest

from anal.Original_Network_Analysis import wnfd_nk as original_calculate_multifractal_taus
from analysis.single_graph_multifractal_analyzer import SingleGraphMultifractalAnalyzer
from config import ConfigSample

ConfigSample.initialize()


@pytest.fixture
def load_graph_from_pickle():
    file_path = os.path.realpath(os.path.join("..", "data", "input_network.pkl"))
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


def test_calculate_multifractal_taus(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    analyzer = SingleGraphMultifractalAnalyzer(sample_graph)

    tau_list, r_g_all, diameter, zq_list = analyzer.calculate_multifractal_taus()
    correct_tau_list, correct_r_g_all, correct_diameter, correct_zq_list = (
        original_calculate_multifractal_taus(sample_graph,
                                             SingleGraphMultifractalAnalyzer.Q,
                                             weight=False))

    assert isinstance(tau_list, list)
    assert isinstance(r_g_all, np.ndarray)
    # assert isinstance(diameter, (float, int))
    assert isinstance(zq_list, list)

    assert len(tau_list) == len(SingleGraphMultifractalAnalyzer.Q)
    assert len(r_g_all) > 0
    assert diameter > 0

    assert len(tau_list) == len(SingleGraphMultifractalAnalyzer.Q)

    assert len(r_g_all) > 0

    assert diameter > 0

    assert np.array_equal(tau_list, correct_tau_list)
    assert np.array_equal(r_g_all, correct_r_g_all)
    assert np.array_equal(diameter, correct_diameter)
    assert np.array_equal(zq_list, correct_zq_list)
