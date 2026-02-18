# tests/analysis/test_multifractal_analyzer.py

import numpy as np
import pytest

from analysis.multifractal_analyzer import MultifractalAnalyzer
from config import AnaConfig, BaseConfig

from .._original_code import (
    original_calculate_assortativity,
    original_calculate_betweenness,
    original_calculate_centralities,
    original_calculate_eigenvector_centrality,
    original_calculate_multifractal_taus,
    original_calculate_ollivier_ricci_curvature,
    original_n_dimension,
    original_n_spectrum,
    original_node_dimension,
)

AnaConfig.initialize()


def test_calculate_multifractal_taus(
    load_unweighted_test_synth_graph, load_unweighted_test_nx_graph
):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph

    analyzer = MultifractalAnalyzer(synth_graph)
    tau_list, zq_list = analyzer._compute_multifractal_taus()

    assert isinstance(tau_list, (np.ndarray, list))
    assert isinstance(zq_list, (np.ndarray, list))

    assert len(tau_list) == len(MultifractalAnalyzer.full_q)

    correct_tau_list, correct_r_g_all, correct_diameter, correct_zq_list = (
        original_calculate_multifractal_taus(
            nx_graph, MultifractalAnalyzer.full_q, weight=False
        )
    )

    assert np.array_equal(tau_list, correct_tau_list)
    assert np.array_equal(zq_list, correct_zq_list)


def test_n_spectrum(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    nx_graph = load_unweighted_test_nx_graph
    synth_graph = load_unweighted_test_synth_graph
    n_tau, _, _, _ = original_calculate_multifractal_taus(
        nx_graph, MultifractalAnalyzer.full_q, weight=False
    )
    analyzer = MultifractalAnalyzer(synth_graph)
    alpha_0, width, al_list, fal_list = analyzer._compute_n_spectrum(n_tau)

    correct_alpha_0, correct_width = original_n_spectrum(
        n_tau, MultifractalAnalyzer.full_q, 0, "b"
    )

    assert np.array_equal(alpha_0, correct_alpha_0)
    assert np.array_equal(width, correct_width)


def test_n_dimension(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    nx_graph = load_unweighted_test_nx_graph
    synth_graph = load_unweighted_test_synth_graph
    Q = MultifractalAnalyzer.full_q
    n_tau, _, _, _ = original_calculate_multifractal_taus(nx_graph, Q, weight=False)
    analyzer = MultifractalAnalyzer(synth_graph)
    dim_list, dim_max, dim_min, diff, valid_q = analyzer._compute_n_dimension(n_tau)

    correct_dim_list, correct_valid_q = original_n_dimension(n_tau, Q, k=0, color="red")

    assert np.array_equal(dim_list, correct_dim_list)
    assert np.array_equal(valid_q, correct_valid_q)


def test_node_dimension(
    load_unweighted_test_synth_graph, load_unweighted_test_nx_graph
):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    analyzer.f_digit = 2
    node_dimension = analyzer._compute_node_dimension()
    correct_node_dimension = original_node_dimension(nx_graph, weight=None)

    assert node_dimension == correct_node_dimension


def test_centralities(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    centralities_dict = analyzer._compute_centralities()
    correct_centralities_dict = original_calculate_centralities(nx_graph, False)

    assert centralities_dict == correct_centralities_dict


def test_betweenness(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    betweenness_dict = analyzer._compute_betweenness()
    correct_betweenness_dict = original_calculate_betweenness(nx_graph, False)

    assert np.allclose(betweenness_dict, correct_betweenness_dict, atol=1e-18)


def test_ricci_curvature(
    load_unweighted_test_synth_graph, load_unweighted_test_nx_graph
):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    ricci_curvature = analyzer._compute_ollivier_ricci_curvature()

    assert isinstance(ricci_curvature, list)
    assert len(ricci_curvature) == synth_graph.number_of_edges()
    assert all(np.isfinite(k) for k in ricci_curvature)

    nx_graph = load_unweighted_test_nx_graph
    grc_curvature = original_calculate_ollivier_ricci_curvature(nx_graph, False)
    assert np.allclose(sorted(ricci_curvature), sorted(grc_curvature), atol=1e-6)


def test_assortativity(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    assortativity = analyzer._compute_assortativity()
    correct_assortativity = original_calculate_assortativity(nx_graph, False)

    assert assortativity == correct_assortativity


def test_eigenvector_centrality(
    load_unweighted_test_synth_graph, load_unweighted_test_nx_graph
):
    synth_graph = load_unweighted_test_synth_graph
    nx_graph = load_unweighted_test_nx_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    eigenvector_centrality = analyzer._compute_eigenvector_centrality()
    correct_eigenvector_centrality = original_calculate_eigenvector_centrality(
        nx_graph, False
    )

    assert eigenvector_centrality == correct_eigenvector_centrality


def test_diameter(load_unweighted_test_synth_graph):
    if not BaseConfig.MEASURE_WEIGHTED:
        return pytest.skip("Test only for weighted graphs")

    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    diameter = analyzer._compute_diameter()

    assert isinstance(diameter, (int, float))
    assert diameter > 0
