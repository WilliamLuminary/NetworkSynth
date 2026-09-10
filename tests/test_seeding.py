# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from networksynth.configs import SynthParams

pytestmark = pytest.mark.unit


BASE = dict(
    synthetic_frame_size=(512, 512),
    closed_nodes_factor=1.2,
    closed_edges_factor=0.8,
    max_attempts=3,
)


def _params(seed):
    return SynthParams(seed=seed, **BASE)


class TestForWorker:
    def test_derives_distinct_seeds(self):
        params = _params(1000)

        seeds = [params.for_worker(i).seed for i in range(5)]

        assert seeds == [1000, 1001, 1002, 1003, 1004]
        assert len(set(seeds)) == 5, "workers must not share a seed"

    def test_is_deterministic(self):
        assert _params(7).for_worker(3).seed == _params(7).for_worker(3).seed

    def test_unseeded_passes_through(self):
        params = _params(None)

        assert params.for_worker(4).seed is None

    def test_leaves_other_fields_alone(self):
        derived = _params(500).for_worker(2)

        assert derived.synthetic_frame_size == BASE["synthetic_frame_size"]
        assert derived.closed_nodes_factor == BASE["closed_nodes_factor"]
        assert derived.max_attempts == BASE["max_attempts"]

    def test_returns_a_new_object(self):
        params = _params(300)

        params.for_worker(9)

        assert params.seed == 300


@pytest.mark.integration
class TestGenerationIsReproducible:

    @staticmethod
    def _generate(seed, attrs):
        """A worker's whole product: the grown network with its weights, both
        drawn from the one Generator the seed opens."""
        from networksynth.graphs.graph_generator import GraphGenerator

        _, attributes, mapper = attrs
        generator = GraphGenerator(attributes, _params(seed))
        graph = generator.generate_network()
        mapper.assign_weights(graph, generator.rng)
        return graph

    @staticmethod
    def _fingerprint(graph):
        return (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(sorted(graph.edges_with_weights())),
            graph.positions().round(6).tobytes(),
        )

    @pytest.fixture
    def attrs(self, load_weighted_test_synth_graph):
        from networksynth.handlers import AttributesCalculator, Mapper

        graph = load_weighted_test_synth_graph
        return graph, AttributesCalculator().analyze(graph), Mapper(graph)

    def test_same_seed_gives_an_identical_network(self, attrs):
        a = self._generate(4242, attrs)
        b = self._generate(4242, attrs)

        assert self._fingerprint(a) == self._fingerprint(b)

    def test_different_seeds_give_different_networks(self, attrs):
        a = self._generate(1, attrs)
        b = self._generate(2, attrs)

        assert self._fingerprint(a) != self._fingerprint(b)

    def test_worker_seeds_produce_distinct_candidates(self, attrs):
        base = _params(777)
        prints = {
            self._fingerprint(self._generate(base.for_worker(i).seed, attrs))
            for i in range(3)
        }

        assert len(prints) == 3
