import pytest

from configs import SynthParams
from graphs._graph_node import GraphNode

pytestmark = pytest.mark.unit


@pytest.fixture
def attributes(load_unweighted_test_synth_graph):
    from handlers import AttributesCalculator

    return AttributesCalculator().analyze(load_unweighted_test_synth_graph)


@pytest.fixture
def params():
    return SynthParams(
        synthetic_frame_size=(200, 200),
        closed_nodes_factor=1.2,
        closed_edges_factor=0.8,
        max_attempts=1,
        seed=1,
    )


@pytest.fixture(autouse=True)
def _no_traversal_left_open():
    yield
    GraphNode._traversal_active = False


class TestTheRecordIsScoped:
    def test_the_parameters_are_installed_inside(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            assert GraphNode._degree_dist is attributes.degree_distribution
            assert GraphNode._closed_nodes_factor == 1.2

    def test_the_grids_are_dropped_on_the_way_out(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            GraphNode((0.0, 0.0))
            assert GraphNode.node_grid, "the traversal recorded nothing"

        # The spatial index over a traversal can hold millions of nodes, so it
        # is freed where the traversal ends rather than at some later point a
        # caller has to remember.
        assert not GraphNode.node_grid
        assert not GraphNode.edge_grid

    def test_the_grids_are_dropped_when_the_body_raises(self, attributes, params):
        with pytest.raises(RuntimeError):
            with GraphNode.traversal(attributes, params):
                GraphNode((0.0, 0.0))
                raise RuntimeError("generation failed")

        assert not GraphNode.node_grid


class TestTwoTraversalsCannotInterleave:
    def test_nesting_is_refused(self, attributes, params):
        # The record is class-level: a second traversal would silently share
        # the first one's grids and node ids.
        with GraphNode.traversal(attributes, params):
            with pytest.raises(AssertionError, match="already open"):
                with GraphNode.traversal(attributes, params):
                    pass

    def test_a_failed_traversal_does_not_block_the_next_one(self, attributes, params):
        with pytest.raises(RuntimeError):
            with GraphNode.traversal(attributes, params):
                raise RuntimeError("generation failed")

        # Nothing to clean up by hand: the scope closed itself.
        with GraphNode.traversal(attributes, params):
            assert GraphNode._traversal_active
