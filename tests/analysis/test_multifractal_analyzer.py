# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest

pytestmark = pytest.mark.requires_fixture_data

from networksynth.analysis.multifractal_analyzer import MultifractalAnalyzer
from tests.fixture_config import FixtureCompareConfig as CompareConfig

CompareConfig.initialize()


def test_calculate_multifractal_taus(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, zq_list = analyzer._compute_multifractal_taus()

    assert isinstance(tau_list, list)
    assert isinstance(zq_list, list)
    assert len(tau_list) == len(MultifractalAnalyzer.full_q)
    assert all(np.isfinite(t) for t in tau_list)


def test_n_spectrum(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, _ = analyzer._compute_multifractal_taus()
        alpha_0, width, al_list, fal_list = analyzer._compute_n_spectrum(tau_list)

    assert np.isfinite(alpha_0)
    assert width >= 0
    assert len(al_list) == len(MultifractalAnalyzer.full_q) - 1
    assert len(fal_list) == len(MultifractalAnalyzer.full_q) - 1


def test_n_dimension(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    with analyzer.set_q(MultifractalAnalyzer.full_q):
        tau_list, _ = analyzer._compute_multifractal_taus()
        dim_list, dim_max, dim_min, diff, valid_q = analyzer._compute_n_dimension(
            tau_list
        )

    assert dim_max >= dim_min
    assert diff == pytest.approx(dim_max - dim_min)
    assert len(dim_list) == len(valid_q)
    assert all(q != 0 for q in valid_q)


def test_node_dimension(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(
        synth_graph, CompareConfig.MEASURE_WEIGHTED, CompareConfig.FULL_Q_BAND
    )
    node_dimension = analyzer._compute_node_dimension()

    assert isinstance(node_dimension, dict)
    assert len(node_dimension) == synth_graph.number_of_nodes()
    assert all(np.isfinite(v) for v in node_dimension.values())


def test_centralities(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(
        synth_graph, CompareConfig.MEASURE_WEIGHTED, CompareConfig.FULL_Q_BAND
    )
    centralities = analyzer._compute_centralities()

    expected_keys = {"nfd", "closeness", "degree", "clustering"}
    assert set(centralities.keys()) == expected_keys
    n = synth_graph.number_of_nodes()
    for key in expected_keys:
        assert len(centralities[key]) == n, f"{key} length mismatch"
        assert all(np.isfinite(v) for v in centralities[key]), f"{key} has non-finite"


def test_betweenness(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    betweenness = analyzer._compute_betweenness()

    assert isinstance(betweenness, list)
    assert len(betweenness) == load_unweighted_test_synth_graph.number_of_nodes()
    assert all(np.isfinite(v) for v in betweenness)
    assert all(0 <= v <= 1 for v in betweenness)


def test_ricci_curvature(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(
        synth_graph, CompareConfig.MEASURE_WEIGHTED, CompareConfig.FULL_Q_BAND
    )
    ricci = analyzer._compute_ollivier_ricci_curvature()

    assert isinstance(ricci, list)
    assert len(ricci) == synth_graph.number_of_edges()
    assert all(np.isfinite(k) for k in ricci)


def test_assortativity(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    assortativity = analyzer._compute_assortativity()

    assert isinstance(assortativity, float)
    assert np.isfinite(assortativity)
    assert -1 <= assortativity <= 1


def test_eigenvector_centrality(load_unweighted_test_synth_graph):
    synth_graph = load_unweighted_test_synth_graph
    analyzer = MultifractalAnalyzer(
        synth_graph, CompareConfig.MEASURE_WEIGHTED, CompareConfig.FULL_Q_BAND
    )
    eigenvector = analyzer._compute_eigenvector_centrality()

    assert isinstance(eigenvector, list)
    assert len(eigenvector) > 0
    assert all(np.isfinite(v) for v in eigenvector)
    assert all(v >= 0 for v in eigenvector)


def test_diameter(load_unweighted_test_synth_graph):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph,
        CompareConfig.MEASURE_WEIGHTED,
        CompareConfig.FULL_Q_BAND,
    )
    diameter = analyzer._compute_diameter()

    assert isinstance(diameter, (int, float))
    assert diameter > 0


def _adjacency(graph):
    from scipy.sparse import coo_matrix

    n = graph.number_of_nodes()
    rows, cols, values = [], [], []
    for u, v, w in graph.edges_with_weights():
        rows += [u, v]
        cols += [v, u]
        values += [w, w]
    return coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


@pytest.mark.parametrize("weighted", [False, True])
def test_eigenvector_centrality_solves_the_eigenproblem(
    weighted, load_weighted_test_synth_graph, load_unweighted_test_synth_graph
):
    graph = (
        load_weighted_test_synth_graph
        if weighted
        else (load_unweighted_test_synth_graph)
    )
    analyzer = MultifractalAnalyzer(graph, measure_weighted=weighted, full_q_band=False)

    scores = np.asarray(analyzer._compute_eigenvector_centrality())
    matrix = _adjacency(analyzer._get_analysis_graph())

    assert len(scores) == matrix.shape[0]
    assert np.isfinite(scores).all()
    assert (scores >= 0).all(), "Perron-Frobenius: single-signed"
    assert np.isclose(np.linalg.norm(scores), 1.0), "unit-L2 by construction"

    eigenvalue = float(scores @ (matrix @ scores))
    residual = np.linalg.norm(matrix @ scores - eigenvalue * scores)
    assert residual < 1e-10, f"not an eigenvector: residual {residual:.2e}"


def test_all_pairs_shortest_paths_are_computed_once_per_graph(
    load_unweighted_test_synth_graph,
):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph, measure_weighted=False, full_q_band=False
    )
    graph = analyzer._get_analysis_graph()

    first = analyzer._get_distances(graph)

    assert analyzer._get_distances(graph) is first, "recomputed instead of cached"


def test_curvature_uses_the_analyzer_graph_not_its_own_copy(
    load_unweighted_test_synth_graph,
):
    analyzer = MultifractalAnalyzer(
        load_unweighted_test_synth_graph, measure_weighted=False, full_q_band=False
    )

    ricci = analyzer._compute_ollivier_ricci_curvature()

    assert len(analyzer._distances) == 1, "curvature ran a second APSP"
    assert len(ricci) == load_unweighted_test_synth_graph.number_of_edges()


class TestNeighbourMasses:

    def test_matches_the_naive_formula_when_it_does_not_underflow(self):
        from networksynth.analysis.multifractal_analyzer import _neighbour_masses

        distances = np.array([0.4, 0.9, 1.3])
        naive_affinity = np.e ** (-(distances**2))
        expected = 0.5 * naive_affinity / naive_affinity.sum()

        masses = _neighbour_masses(distances, alpha=0.5, base=np.e, exp_power=2)

        assert np.allclose(masses, expected, rtol=0, atol=1e-15)

    def test_survives_distances_that_underflow(self):
        from networksynth.analysis.multifractal_analyzer import _neighbour_masses

        distances = np.array([30.0, 40.0, 50.0])
        assert (np.e ** (-(distances**2)) == 0).all(), "premise: these underflow"

        masses = _neighbour_masses(distances, alpha=0.5, base=np.e, exp_power=2)

        assert np.isfinite(masses).all()
        assert np.isclose(masses.sum(), 0.5), "must still carry 1 - alpha"

    def test_nearest_neighbour_takes_the_most_mass(self):
        from networksynth.analysis.multifractal_analyzer import _neighbour_masses

        masses = _neighbour_masses(
            np.array([30.0, 31.0, 32.0]), alpha=0.5, base=np.e, exp_power=2
        )

        assert masses[0] > masses[1] > masses[2]

    def test_weighted_curvature_completes(self, load_weighted_test_synth_graph):
        graph = load_weighted_test_synth_graph
        analyzer = MultifractalAnalyzer(graph, measure_weighted=True, full_q_band=False)

        ricci = analyzer._compute_ollivier_ricci_curvature()

        assert len(ricci) == graph.number_of_edges()
        assert np.isfinite(ricci).all()
