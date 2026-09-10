# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from networksynth.configs import GenerateConfig, SynthParams

pytestmark = pytest.mark.unit


def _config():
    return GenerateConfig(
        DATASETS=[],
        FRAME_SIZE=(512, 512),
        SYNTHETIC_FRAME_SIZE=(512, 512),
        CLOSED_NODES_FACTOR=1.2,
        CLOSED_EDGES_FACTOR=0.8,
        MAX_ATTEMPTS=7,
        MEASURE_WEIGHTED=False,
    )


class TestFromConfig:
    def test_reads_all_fields(self):
        params = SynthParams.from_config(_config())

        assert params.synthetic_frame_size == (512, 512)
        assert params.closed_nodes_factor == 1.2
        assert params.closed_edges_factor == 0.8
        assert params.max_attempts == 7

    def test_accepts_an_explicit_config(self):
        class Cfg:
            SYNTHETIC_FRAME_SIZE = (64, 64)
            CLOSED_NODES_FACTOR = 2.0
            CLOSED_EDGES_FACTOR = 3.0
            MAX_ATTEMPTS = 1
            SEED = 123

        params = SynthParams.from_config(Cfg)

        assert params.synthetic_frame_size == (64, 64)
        assert params.closed_nodes_factor == 2.0
        assert params.closed_edges_factor == 3.0
        assert params.max_attempts == 1
        assert params.seed == 123


class TestImmutability:
    def test_is_frozen(self):
        params = SynthParams((10, 10), 1.0, 1.0, 5, None)

        with pytest.raises(Exception):
            params.max_attempts = 99

    def test_survives_a_pickle_round_trip(self):
        import pickle

        params = SynthParams((128, 256), 1.5, 0.5, max_attempts=3, seed=42)

        assert pickle.loads(pickle.dumps(params)) == params


class TestGenerationReadsParamsNotAConfig:

    def test_the_traversal_rules_come_from_params(self):
        from networksynth.graphs._graph_node import Traversal

        class Attrs:
            degree_distribution = {2: 1.0}
            degree_transition_probs = {2: {2: 1.0}}
            degree_angles = {2: [0.0]}
            degree_lengths = {2: [10.0]}
            average_length = 10.0

        params = SynthParams(
            (512, 512),
            closed_nodes_factor=5.0,
            closed_edges_factor=4.0,
            max_attempts=1,
            seed=None,
        )
        rules = Traversal.build(Attrs(), params, params.rng()).rules

        assert rules.closed_nodes_thr_sq == (10.0 * 5.0) ** 2
        assert rules.closed_edges_thr_sq == (10.0 * 4.0) ** 2

    def test_generator_uses_params_frame_size(self):
        from networksynth.graphs.graph_generator import GraphGenerator

        class Attrs:
            degree_distribution = {2: 1.0}
            degree_transition_probs = {2: {2: 1.0}}
            degree_angles = {2: [0.0]}
            degree_lengths = {2: [10.0]}
            average_length = 10.0
            average_degree = 2.0

        params = SynthParams((999, 777), 1.2, 0.8, 3, None)
        generator = GraphGenerator(Attrs(), params)

        assert generator._params.synthetic_frame_size == (999, 777)

    def test_generator_requires_params(self):
        from networksynth.graphs.graph_generator import GraphGenerator

        class Attrs:
            degree_distribution = {2: 1.0}
            degree_transition_probs = {2: {2: 1.0}}
            degree_angles = {2: [0.0]}
            degree_lengths = {2: [10.0]}
            average_length = 10.0
            average_degree = 2.0

        with pytest.raises(TypeError):
            GraphGenerator(Attrs())


class TestNoDefaults:

    def test_every_field_is_required(self):
        with pytest.raises(TypeError):
            SynthParams()
        with pytest.raises(TypeError):
            SynthParams((10, 10), 1.0, 1.0)

    def test_from_config_requires_a_config(self):
        with pytest.raises(TypeError):
            SynthParams.from_config()

    def test_incomplete_config_raises(self):
        class Incomplete:
            SYNTHETIC_FRAME_SIZE = (10, 10)
            CLOSED_NODES_FACTOR = 1.0

        with pytest.raises(AttributeError):
            SynthParams.from_config(Incomplete)
