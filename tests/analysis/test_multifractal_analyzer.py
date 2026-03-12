# tests/analysis/test_multifractal_analyzer.py

import numpy as np
import pytest

from analysis.multifractal_analyzer import MultifractalAnalyzer
from configs.analyze_mode.config_sample import SampleConfig as AnaConfig

AnaConfig.initialize()

from .._original_code import (
    original_calculate_assortativity,
    original_calculate_betweenness,
    original_calculate_ollivier_ricci_curvature,
)


@pytest.fixture(autouse=True)
def _ensure_ana_config():
    """Re-inject AnaConfig so other modules' initialize() can't pollute state."""
    AnaConfig._inject_dependencies()


# ---------------------------------------------------------------------------
# Multifractal core (property-based, no slow reference)
# ---------------------------------------------------------------------------


def test_calculate_multifractal_taus(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, zq_list = analyzer._compute_multifractal_taus()

    assert isinstance(tau_list, list)
    assert isinstance(zq_list, list)
    assert len(tau_list) == len(MultifractalAnalyzer.full_q)
    assert all(np.isfinite(t) for t in tau_list)


def test_n_spectrum(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, _ = analyzer._compute_multifractal_taus()
        alpha_0, width, al_list, fal_list = analyzer._compute_n_spectrum(tau_list)

    assert np.isfinite(alpha_0)
    assert width >= 0
    assert len(al_list) == len(MultifractalAnalyzer.full_q) - 1
    assert len(fal_list) == len(MultifractalAnalyzer.full_q) - 1


def test_n_dimension(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, _ = analyzer._compute_multifractal_taus()
        dim_list, dim_max, dim_min, diff, valid_q = analyzer._compute_n_dimension(
            tau_list
        )

    assert dim_max >= dim_min
    assert diff == pytest.approx(dim_max - dim_min)
    assert len(dim_list) == len(valid_q)
    assert all(q != 0 for q in valid_q)


# ---------------------------------------------------------------------------
# Node dimension & centralities (property-based)
# ---------------------------------------------------------------------------


def test_node_dimension(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    node_dimension = analyzer._compute_node_dimension()

    assert isinstance(node_dimension, dict)
    assert len(node_dimension) == synth_graph.number_of_nodes()
    assert all(np.isfinite(v) for v in node_dimension.values())


def test_centralities(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    centralities = analyzer._compute_centralities()

    expected_keys = {"nfd", "closeness", "degree", "clustering"}
    assert set(centralities.keys()) == expected_keys
    n = synth_graph.number_of_nodes()
    for key in expected_keys:
        assert len(centralities[key]) == n, f"{key} length mismatch"
        assert all(np.isfinite(v) for v in centralities[key]), f"{key} has non-finite"


# ---------------------------------------------------------------------------
# Betweenness (compare with reference — both use networkit, fast)
# ---------------------------------------------------------------------------


def test_betweenness(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    betweenness = analyzer._compute_betweenness()
    correct = original_calculate_betweenness(load_unweighted_test_nx_graph, False)

    assert np.allclose(betweenness, correct, atol=1e-6)


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature (property-based + cross-check with native ref)
# ---------------------------------------------------------------------------


def test_ricci_curvature(
    load_unweighted_test_synth_graph, load_unweighted_test_nx_graph
):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    ricci = analyzer._compute_ollivier_ricci_curvature()

    assert isinstance(ricci, list)
    assert len(ricci) == synth_graph.number_of_edges()
    assert all(np.isfinite(k) for k in ricci)

    ref_ricci = original_calculate_ollivier_ricci_curvature(
        load_unweighted_test_nx_graph, False
    )
    assert np.allclose(sorted(ricci), sorted(ref_ricci), atol=1e-6)


# ---------------------------------------------------------------------------
# Assortativity (compare with nx reference — fast)
# ---------------------------------------------------------------------------


def test_assortativity(load_unweighted_test_synth_graph, load_unweighted_test_nx_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    assortativity = analyzer._compute_assortativity()
    correct = original_calculate_assortativity(load_unweighted_test_nx_graph, False)

    assert assortativity == pytest.approx(correct, abs=1e-6)


# ---------------------------------------------------------------------------
# Eigenvector centrality (property-based — nk vs nx normalisation differs)
# ---------------------------------------------------------------------------


def test_eigenvector_centrality(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(synth_graph)
    eigenvector = analyzer._compute_eigenvector_centrality()

    assert isinstance(eigenvector, list)
    assert len(eigenvector) > 0
    assert all(np.isfinite(v) for v in eigenvector)
    assert all(v >= 0 for v in eigenvector)


# ---------------------------------------------------------------------------
# Diameter
# ---------------------------------------------------------------------------


def test_diameter(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(load_unweighted_test_synth_graph)
    diameter = analyzer._compute_diameter()

    assert isinstance(diameter, (int, float))
    assert diameter > 0
