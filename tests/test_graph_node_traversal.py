# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from configs import SynthParams
from graphs._graph_node import GraphNode, PlacementCounts

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
            assert GraphNode._rules.degree_dist is attributes.degree_distribution
            assert GraphNode._rules.grid_size == attributes.average_length

    def test_the_grids_are_dropped_on_the_way_out(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            GraphNode((0.0, 0.0))
            assert GraphNode.node_grid, "the traversal recorded nothing"

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
        with GraphNode.traversal(attributes, params):
            with pytest.raises(AssertionError, match="already open"):
                with GraphNode.traversal(attributes, params):
                    pass

    def test_a_failed_traversal_does_not_block_the_next_one(self, attributes, params):
        with pytest.raises(RuntimeError):
            with GraphNode.traversal(attributes, params):
                raise RuntimeError("generation failed")

        with GraphNode.traversal(attributes, params):
            assert GraphNode._traversal_active


class TestTheRulesAreSeparateFromTheRecord:
    def test_a_reset_keeps_the_rules(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            rules = GraphNode._rules
            GraphNode((0.0, 0.0))

            GraphNode.reset()

            assert GraphNode._rules is rules
            assert not GraphNode.node_grid

    def test_the_rules_are_immutable(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            with pytest.raises(Exception):
                GraphNode._rules.grid_size = 1.0


class TestPlacementCountsAreAResult:
    def test_they_start_at_zero_and_read_back(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            assert GraphNode.counts() == PlacementCounts(merged=0, aborted=0)

            GraphNode._merged_edge = 3
            GraphNode._aborted_edge = 4

            assert GraphNode.counts() == PlacementCounts(merged=3, aborted=4)

    def test_they_format_themselves_for_a_log_line(self):
        assert str(PlacementCounts(merged=1234, aborted=56)) == (
            "merged=1,234, aborted=56"
        )

    def test_a_reset_clears_them(self, attributes, params):
        with GraphNode.traversal(attributes, params):
            GraphNode._merged_edge = 9
            GraphNode.reset()

            assert GraphNode.counts().merged == 0
