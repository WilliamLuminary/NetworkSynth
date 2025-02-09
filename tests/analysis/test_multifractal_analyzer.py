import os
import pickle

import networkx as nx
import numpy as np
import pytest

from anal.Original_Network_Analysis import nspectrum as original_n_spectrum
from anal.Original_Network_Analysis import wnfd_nk as original_calculate_multifractal_taus
from anal.Original_Network_Analysis import ndimension as original_n_dimension
from anal.Original_Network_Analysis import node_dimension as original_node_dimension
from analysis.single_graph_multifractal_analyzer import SingleGraphMultifractalAnalyzer
from config import Config, ConfigSample

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
    tau_list, r_g_all, diameter, zq_list = analyzer.compute_multifractal_taus()

    assert isinstance(tau_list, (np.ndarray, list))
    assert isinstance(r_g_all, (np.ndarray, list))
    assert isinstance(diameter, (np.int64, np.int32, int, float))
    assert isinstance(zq_list, (np.ndarray, list))

    assert len(tau_list) == len(SingleGraphMultifractalAnalyzer.Q)
    assert len(r_g_all) > 0
    assert diameter > 0

    correct_tau_list, correct_r_g_all, correct_diameter, correct_zq_list = (
        original_calculate_multifractal_taus(sample_graph,
                                             SingleGraphMultifractalAnalyzer.Q,
                                             weight=False))

    assert np.array_equal(tau_list, correct_tau_list)
    assert np.array_equal(r_g_all, correct_r_g_all)
    assert np.array_equal(diameter, correct_diameter)
    assert np.array_equal(zq_list, correct_zq_list)


def test_n_spectrum(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    ntau, _, _, _ = original_calculate_multifractal_taus(sample_graph, SingleGraphMultifractalAnalyzer.Q, weight=False)
    analyzer = SingleGraphMultifractalAnalyzer(sample_graph)
    alpha_0, width, al_list, fal_list = analyzer.compute_n_spectrum(ntau)

    correct_alpha_0, correct_width = original_n_spectrum(ntau, SingleGraphMultifractalAnalyzer.Q, 0, 'b')

    assert np.array_equal(alpha_0, correct_alpha_0)
    assert np.array_equal(width, correct_width)
    # assert np.array_equal(max_al_val, correct_al_list)
    # assert np.array_equal(min_al_val, correct_fal_list)


def test_n_dimension(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    Q = SingleGraphMultifractalAnalyzer.Q
    ntau, _, _, _ = original_calculate_multifractal_taus(sample_graph, Q, weight=False)
    analyzer = SingleGraphMultifractalAnalyzer(sample_graph)
    dim_list, dim_max, dim_min, diff, valid_q = analyzer.compute_n_dimension(ntau, label=0, color="red")

    correct_dim_list, correct_valid_q = original_n_dimension(ntau, Q, k=0, color="red")

    assert np.array_equal(dim_list, correct_dim_list)
    assert np.array_equal(valid_q, correct_valid_q)

def test_node_dimension(load_graph_from_pickle):
    sample_graph = load_graph_from_pickle
    analyzer = SingleGraphMultifractalAnalyzer(sample_graph)
    analyzer.f_digit = 2
    node_dimension = analyzer.compute_node_dimension()

    correct_node_dimension = original_node_dimension(sample_graph, weight=None)

    assert node_dimension == correct_node_dimension
