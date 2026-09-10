# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from networksynth.configs import SynthParams
from networksynth.graphs._graph_node import GraphNode, PlacementCounts, Traversal

pytestmark = pytest.mark.unit


@pytest.fixture
def attributes(load_unweighted_test_synth_graph):
    from networksynth.handlers import AttributesCalculator

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


class TestATraversalOwnsItsRecord:
    def test_the_rules_come_from_the_attributes(self, attributes, params):
        traversal = Traversal.build(attributes, params, params.rng())

        assert traversal.rules.degree_dist is attributes.degree_distribution
        assert traversal.rules.grid_size == attributes.average_length

    def test_the_rules_are_immutable(self, attributes, params):
        with pytest.raises(Exception):
            Traversal.build(attributes, params, params.rng()).rules.grid_size = 1.0

    def test_a_node_is_filed_in_the_traversal_it_was_grown_in(self, attributes, params):
        traversal = Traversal.build(attributes, params, params.rng())

        GraphNode(traversal, (0.0, 0.0))

        assert traversal.node_grid, "the traversal recorded nothing"
        assert traversal.edge_grid

    def test_two_traversals_do_not_share_a_grid(self, attributes, params):
        first = Traversal.build(attributes, params, params.rng())
        second = Traversal.build(attributes, params, params.rng())

        GraphNode(first, (0.0, 0.0))

        assert not second.node_grid
        assert not second.edge_grid
        assert second.nodes_created() == 0

    def test_ids_count_from_zero_per_traversal(self, attributes, params):
        first = Traversal.build(attributes, params, params.rng())
        second = Traversal.build(attributes, params, params.rng())

        assert [first.next_id(), first.next_id()] == [0, 1]
        assert second.next_id() == 0
        assert first.nodes_created() == 2


class TestPlacementCountsAreAResult:
    def test_they_start_at_zero_and_read_back(self, attributes, params):
        traversal = Traversal.build(attributes, params, params.rng())
        assert traversal.counts() == PlacementCounts(merged=0, aborted=0)

        traversal.merged = 3
        traversal.aborted = 4

        assert traversal.counts() == PlacementCounts(merged=3, aborted=4)

    def test_they_format_themselves_for_a_log_line(self):
        assert str(PlacementCounts(merged=1234, aborted=56)) == (
            "merged=1,234, aborted=56"
        )


class TestGenerationLeavesNothingBehind:
    def test_a_generator_holds_no_grid_after_generating(self, attributes, params):
        from networksynth.graphs import GraphGenerator

        generator = GraphGenerator(attributes, params)
        generator.generate_network()

        assert not any(
            isinstance(value, Traversal) for value in vars(generator).values()
        )
        assert not hasattr(GraphNode, "node_grid")

    def test_two_generators_keep_their_own_rules(self, attributes, params):
        from dataclasses import replace

        from networksynth.graphs import GraphGenerator

        loose = GraphGenerator(attributes, replace(params, closed_nodes_factor=5.0))
        tight = GraphGenerator(attributes, params)

        assert loose._params.closed_nodes_factor == 5.0
        assert tight._params.closed_nodes_factor == 1.2
