"""Reproducibility guarantees of BaseConfig.RANDOM_SEED.

Generation is stochastic, so the only thing that makes a published run
verifiable is that re-seeding with the same value replays it exactly.
These tests pin that contract:

  - same RANDOM_SEED + same worker index  -> byte-identical network
  - same RANDOM_SEED + different index    -> different network
    (otherwise a parallel batch would collapse into N copies of one network)
  - RANDOM_SEED = None                    -> opt out, runs diverge
"""

import pytest

pytestmark = pytest.mark.integration


def _fingerprint(graph):
    """Positions + sorted edge set, as a comparable tuple."""
    positions = graph.positions()
    edges = sorted(graph.edges())
    return positions.tobytes(), edges


@pytest.fixture(scope="module")
def attributes():
    from configs.generate_mode.config_sample import SampleConfig

    SampleConfig.DATASETS = SampleConfig.DATASETS[:1]
    SampleConfig.initialize()

    from configs import BaseConfig
    from handlers import RunAgent, Saver

    Saver.initialize()
    agent = RunAgent(dataset_id=BaseConfig.get_datasets()[0])
    agent.prepare_data()
    return agent.attributes


def _generate(attributes, seed, index):
    from configs import BaseConfig
    from graphs import GraphGenerator

    original_seed = BaseConfig.RANDOM_SEED
    try:
        BaseConfig.RANDOM_SEED = seed
        BaseConfig.seed_rng(index)
        return _fingerprint(GraphGenerator(attributes).generate_network())
    finally:
        BaseConfig.RANDOM_SEED = original_seed


class TestRandomSeed:
    def test_same_seed_reproduces_network(self, attributes):
        first = _generate(attributes, seed=42, index=0)
        second = _generate(attributes, seed=42, index=0)
        assert first == second

    def test_different_index_diverges(self, attributes):
        first = _generate(attributes, seed=42, index=0)
        second = _generate(attributes, seed=42, index=1)
        assert first != second

    def test_different_seed_diverges(self, attributes):
        first = _generate(attributes, seed=42, index=0)
        second = _generate(attributes, seed=1234, index=0)
        assert first != second

    def test_none_seed_leaves_rng_untouched(self, attributes):
        """RANDOM_SEED = None must not seed, so runs are free to diverge."""
        import random

        from configs import BaseConfig

        original_seed = BaseConfig.RANDOM_SEED
        try:
            BaseConfig.RANDOM_SEED = None
            random.seed(7)
            expected = random.random()

            random.seed(7)
            BaseConfig.seed_rng(0)
            assert random.random() == expected
        finally:
            BaseConfig.RANDOM_SEED = original_seed
