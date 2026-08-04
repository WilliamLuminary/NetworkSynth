# tests/test_synth_params.py
"""
Unit tests for SynthParams (configs/params.py).

The point of SynthParams is that worker processes receive their parameters
explicitly instead of inheriting a mutated ``BaseConfig`` — which only reaches
children under a ``fork`` start method.  These tests pin that guarantee by
giving BaseConfig deliberately wrong values and asserting the passed-in params
win.
"""

import pytest

from configs import BaseConfig, SynthParams

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config_values():
    """Set known BaseConfig values, restoring whatever was there before."""
    saved = {
        name: getattr(BaseConfig, name, None)
        for name in (
            "SYNTHETIC_FRAME_SIZE",
            "CLOSED_NODES_FACTOR",
            "CLOSED_EDGES_FACTOR",
            "MAX_ATTEMPTS",
        )
    }
    BaseConfig.SYNTHETIC_FRAME_SIZE = (512, 512)
    BaseConfig.CLOSED_NODES_FACTOR = 1.2
    BaseConfig.CLOSED_EDGES_FACTOR = 0.8
    BaseConfig.MAX_ATTEMPTS = 7
    yield
    for name, value in saved.items():
        if value is None:
            if hasattr(BaseConfig, name):
                delattr(BaseConfig, name)
        else:
            setattr(BaseConfig, name, value)


class TestFromConfig:
    def test_reads_all_fields(self, base_config_values):
        params = SynthParams.from_config(BaseConfig)

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

        params = SynthParams.from_config(Cfg)

        assert params.synthetic_frame_size == (64, 64)
        assert params.closed_nodes_factor == 2.0
        assert params.closed_edges_factor == 3.0
        assert params.max_attempts == 1


class TestImmutability:
    def test_is_frozen(self):
        params = SynthParams((10, 10), 1.0, 1.0, 5)

        with pytest.raises(Exception):
            params.max_attempts = 99

    def test_survives_a_pickle_round_trip(self):
        """Workers receive params by pickle under a spawn start method."""
        import pickle

        params = SynthParams((128, 256), 1.5, 0.5, max_attempts=3)

        assert pickle.loads(pickle.dumps(params)) == params


class TestParamsWinOverBaseConfig:
    """The spawn-safety guarantee: passed-in params must be authoritative."""

    def test_graph_node_uses_params_not_base_config(self, base_config_values):
        from graphs._graph_node import GraphNode

        class Attrs:
            degree_distribution = {2: 1.0}
            degree_transition_probs = {2: {2: 1.0}}
            degree_angles = {2: [0.0]}
            degree_lengths = {2: [10.0]}
            average_length = 10.0

        # BaseConfig says 1.2 / 0.8; params say something else entirely.
        params = SynthParams(
            (512, 512), closed_nodes_factor=5.0, closed_edges_factor=4.0, max_attempts=1
        )
        GraphNode.initialize(Attrs(), params)

        assert GraphNode._closed_nodes_factor == 5.0
        assert GraphNode._closed_edges_factor == 4.0
        assert GraphNode._closed_nodes_thr == 10.0 * 5.0
        assert GraphNode._closed_edges_thr == 10.0 * 4.0

    def test_generator_uses_params_frame_size(self, base_config_values):
        from graphs.graph_generator import GraphGenerator

        class Attrs:
            degree_distribution = {2: 1.0}
            degree_transition_probs = {2: {2: 1.0}}
            degree_angles = {2: [0.0]}
            degree_lengths = {2: [10.0]}
            average_length = 10.0
            average_degree = 2.0

        params = SynthParams((999, 777), 1.2, 0.8, 3)
        generator = GraphGenerator(Attrs(), params)

        assert generator._params.synthetic_frame_size == (999, 777)

    def test_generator_requires_params(self):
        """Omitting params must fail immediately, not silently fall back."""
        from graphs.graph_generator import GraphGenerator

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
    """A misconfigured run must crash rather than proceed on a fallback."""

    def test_every_field_is_required(self):
        with pytest.raises(TypeError):
            SynthParams()
        with pytest.raises(TypeError):
            SynthParams((10, 10), 1.0, 1.0)  # missing max_attempts

    def test_from_config_requires_a_config(self):
        with pytest.raises(TypeError):
            SynthParams.from_config()

    def test_incomplete_config_raises(self):
        class Incomplete:
            SYNTHETIC_FRAME_SIZE = (10, 10)
            CLOSED_NODES_FACTOR = 1.0
            # CLOSED_EDGES_FACTOR and MAX_ATTEMPTS missing

        with pytest.raises(AttributeError):
            SynthParams.from_config(Incomplete)
