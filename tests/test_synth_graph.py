# tests/test_synth_graph.py
"""
Unit tests for the SynthGraph class in graphs/synth_graph.py.

Covers:
  - Core properties (number_of_nodes, number_of_edges, is_weighted)
  - Position access (positions, position, set_position)
  - Degree helpers (degree, degrees, degree_sequence, weighted_degree)
  - Edge/weight helpers (weight, set_weight, has_edge, remove_edge)
  - Iteration (nodes, neighbors, edges, edges_with_weights)
  - Structural queries (is_connected, largest_connected_component, subgraph, copy)
  - Weight conversions (make_weighted, make_unweighted)
  - Factory methods (from_sparse_matrix, from_edge_list)
  - __repr__
"""

import networkit as nk
import numpy as np
import pytest

pytestmark = pytest.mark.unit

from graphs.synth_graph import SynthGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_graph(n: int = 5, weighted: bool = False) -> SynthGraph:
    """Create a simple chain graph: 0-1-2-..-(n-1)."""
    g = nk.Graph(n, weighted=weighted)
    for i in range(n - 1):
        if weighted:
            g.addEdge(i, i + 1, float(i + 1))
        else:
            g.addEdge(i, i + 1)
    positions = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    return SynthGraph(g, positions)


def _make_disconnected_graph() -> SynthGraph:
    """Create a graph with two components: {0-1-2} and {3-4}."""
    g = nk.Graph(5, weighted=False)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    g.addEdge(3, 4)
    positions = np.array([[0, 0], [1, 0], [2, 0], [10, 10], [11, 10]], dtype=np.float64)
    return SynthGraph(g, positions)


# ---------------------------------------------------------------------------
# Core properties
# ---------------------------------------------------------------------------


class TestCoreProperties:
    def test_number_of_nodes(self):
        g = _make_chain_graph(5)
        assert g.number_of_nodes() == 5

    def test_number_of_edges(self):
        g = _make_chain_graph(5)
        assert g.number_of_edges() == 4

    def test_is_weighted_false(self):
        g = _make_chain_graph(5, weighted=False)
        assert g.is_weighted() is False

    def test_is_weighted_true(self):
        g = _make_chain_graph(5, weighted=True)
        assert g.is_weighted() is True

    def test_nk_property(self):
        g = _make_chain_graph(3)
        assert isinstance(g.nk, nk.Graph)


# ---------------------------------------------------------------------------
# Position access
# ---------------------------------------------------------------------------


class TestPositionAccess:
    def test_positions_shape(self):
        g = _make_chain_graph(5)
        assert g.positions().shape == (5, 2)

    def test_single_position(self):
        g = _make_chain_graph(3)
        pos = g.position(0)
        np.testing.assert_array_equal(pos, [0.0, 1.0])

    def test_set_position(self):
        g = _make_chain_graph(3)
        g.set_position(0, [99.0, 88.0])
        np.testing.assert_array_equal(g.position(0), [99.0, 88.0])


# ---------------------------------------------------------------------------
# Degree helpers
# ---------------------------------------------------------------------------


class TestDegreeHelpers:
    def test_degree_endpoints(self):
        g = _make_chain_graph(5)
        assert g.degree(0) == 1  # endpoint
        assert g.degree(2) == 2  # middle

    def test_degrees(self):
        g = _make_chain_graph(4)
        degs = g.degrees()
        assert len(degs) == 4
        assert degs[0] == (0, 1)
        assert degs[1] == (1, 2)

    def test_degree_sequence(self):
        g = _make_chain_graph(4)
        seq = g.degree_sequence()
        assert seq == [1, 2, 2, 1]

    def test_weighted_degree(self):
        g = _make_chain_graph(3, weighted=True)
        # edge 0-1 has weight 1, edge 1-2 has weight 2
        assert g.weighted_degree(1) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Edge / weight helpers
# ---------------------------------------------------------------------------


class TestEdgeWeightHelpers:
    def test_weight(self):
        g = _make_chain_graph(3, weighted=True)
        assert g.weight(0, 1) == pytest.approx(1.0)
        assert g.weight(1, 2) == pytest.approx(2.0)

    def test_set_weight(self):
        g = _make_chain_graph(3, weighted=True)
        g.set_weight(0, 1, 42.0)
        assert g.weight(0, 1) == pytest.approx(42.0)

    def test_has_edge(self):
        g = _make_chain_graph(4)
        assert g.has_edge(0, 1) is True
        assert g.has_edge(0, 3) is False

    def test_remove_edge(self):
        g = _make_chain_graph(4)
        g.remove_edge(1, 2)
        assert g.has_edge(1, 2) is False
        assert g.number_of_edges() == 2


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_nodes(self):
        g = _make_chain_graph(4)
        node_list = list(g.nodes())
        assert sorted(node_list) == [0, 1, 2, 3]

    def test_neighbors(self):
        g = _make_chain_graph(4)
        assert sorted(g.neighbors(1)) == [0, 2]
        assert g.neighbors(0) == [1]

    def test_edges(self):
        g = _make_chain_graph(3)
        edges = list(g.edges())
        assert len(edges) == 2

    def test_edges_with_weights(self):
        g = _make_chain_graph(3, weighted=True)
        eww = list(g.edges_with_weights())
        assert len(eww) == 2
        # Each entry is (u, v, w)
        weights = {w for _, _, w in eww}
        assert 1.0 in weights
        assert 2.0 in weights


# ---------------------------------------------------------------------------
# Structural queries
# ---------------------------------------------------------------------------


class TestStructuralQueries:
    def test_is_connected_true(self):
        g = _make_chain_graph(4)
        assert g.is_connected() is True

    def test_is_connected_false(self):
        g = _make_disconnected_graph()
        assert g.is_connected() is False

    def test_largest_connected_component(self):
        g = _make_disconnected_graph()
        lcc = g.largest_connected_component()
        assert lcc.number_of_nodes() == 3  # {0, 1, 2}
        assert lcc.is_connected() is True

    def test_lcc_single_component_unchanged(self):
        g = _make_chain_graph(4)
        lcc = g.largest_connected_component()
        assert lcc.number_of_nodes() == 4

    def test_subgraph(self):
        g = _make_chain_graph(5)
        sub = g.subgraph({1, 2, 3})
        assert sub.number_of_nodes() == 3
        # edges 1-2 and 2-3 should be preserved
        assert sub.number_of_edges() == 2

    def test_copy(self):
        g = _make_chain_graph(4)
        g2 = g.copy()
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()
        # Mutation of copy should not affect original
        g2.remove_edge(0, 1)
        assert g.has_edge(0, 1) is True


# ---------------------------------------------------------------------------
# Weight conversions
# ---------------------------------------------------------------------------


class TestWeightConversions:
    def test_make_weighted(self):
        g = _make_chain_graph(3, weighted=False)
        assert g.is_weighted() is False
        g.make_weighted()
        assert g.is_weighted() is True
        assert g.weight(0, 1) == pytest.approx(1.0)

    def test_make_weighted_noop(self):
        g = _make_chain_graph(3, weighted=True)
        g.make_weighted()  # should not raise
        assert g.is_weighted() is True

    def test_make_unweighted(self):
        g = _make_chain_graph(3, weighted=True)
        g.make_unweighted()
        assert g.is_weighted() is False

    def test_make_unweighted_noop(self):
        g = _make_chain_graph(3, weighted=False)
        g.make_unweighted()  # should not raise
        assert g.is_weighted() is False


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    def test_from_sparse_matrix(self):
        from scipy.sparse import csr_matrix

        n = 4
        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        row = [0, 1, 2, 1, 2, 3]
        col = [1, 0, 1, 2, 3, 2]
        data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        mat = csr_matrix((data, (row, col)), shape=(n, n))
        g = SynthGraph.from_sparse_matrix(positions, mat)
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 3
        assert g.is_weighted() is True

    def test_from_edge_list(self):
        positions = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]])
        g = SynthGraph.from_edge_list(positions, edges)
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2
        assert g.is_weighted() is False


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_format(self):
        g = _make_chain_graph(3)
        r = repr(g)
        assert "SynthGraph" in r
        assert "nodes=3" in r
        assert "edges=2" in r
        assert "weighted=False" in r


# ---------------------------------------------------------------------------
# Weight validity
# ---------------------------------------------------------------------------


class TestWeightsMustBePositive:
    """An edge with no width is not a measurable graph.

    Analysis inverts weights into distances, so a zero becomes an infinite
    distance that silently detaches the edge and only surfaces much later as a
    NaN inside the curvature solver. Both paths a weight can enter through
    reject it at the edge that carries it.
    """

    @staticmethod
    def _matrix(values):
        from scipy.sparse import coo_matrix

        rows = [0, 1, 1, 2]
        cols = [1, 0, 2, 1]
        return coo_matrix((values, (rows, cols)), shape=(3, 3))

    @staticmethod
    def _positions():
        return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    def test_set_weight_rejects_zero(self):
        graph = SynthGraph.from_sparse_matrix(
            self._positions(), self._matrix([1.5, 1.5, 2.0, 2.0])
        )

        with pytest.raises(AssertionError, match="non-positive weight"):
            graph.set_weight(0, 1, 0.0)

    def test_set_weight_rejects_negative(self):
        graph = SynthGraph.from_sparse_matrix(
            self._positions(), self._matrix([1.5, 1.5, 2.0, 2.0])
        )

        with pytest.raises(AssertionError, match="non-positive weight"):
            graph.set_weight(0, 1, -1.0)

    def test_set_weight_names_the_offending_edge(self):
        """The Mapper writes one edge at a time; the message must localise it."""
        graph = SynthGraph.from_sparse_matrix(
            self._positions(), self._matrix([1.5, 1.5, 2.0, 2.0])
        )

        with pytest.raises(AssertionError, match=r"edge \(1, 2\)"):
            graph.set_weight(1, 2, 0.0)

    def test_set_weight_accepts_positive(self):
        graph = SynthGraph.from_sparse_matrix(
            self._positions(), self._matrix([1.5, 1.5, 2.0, 2.0])
        )

        graph.set_weight(0, 1, 2.5)

        assert graph.weight(0, 1) == 2.5

    def test_from_sparse_matrix_rejects_an_explicitly_stored_zero(self):
        """Sparse formats can store a zero, so the bad state is representable."""
        with pytest.raises(AssertionError, match="non-positive weight"):
            SynthGraph.from_sparse_matrix(
                self._positions(), self._matrix([1.5, 1.5, 0.0, 0.0])
            )
