import pytest

from configs import BaseConfig, SynthParams

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config_values():
    saved = {
        name: getattr(BaseConfig, name, None)
        for name in (
            "SYNTHETIC_FRAME_SIZE",
            "CLOSED_NODES_FACTOR",
            "CLOSED_EDGES_FACTOR",
            "MAX_ATTEMPTS",
            "SEED",
        )
    }
    BaseConfig.SYNTHETIC_FRAME_SIZE = (512, 512)
    BaseConfig.CLOSED_NODES_FACTOR = 1.2
    BaseConfig.CLOSED_EDGES_FACTOR = 0.8
    BaseConfig.MAX_ATTEMPTS = 7
    BaseConfig.SEED = None
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


class TestParamsWinOverBaseConfig:

    def test_graph_node_uses_params_not_base_config(self, base_config_values):
        from graphs._graph_node import GraphNode

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
        GraphNode.initialize(Attrs(), params)

        assert GraphNode._rules.closed_nodes_thr_sq == (10.0 * 5.0) ** 2
        assert GraphNode._rules.closed_edges_thr_sq == (10.0 * 4.0) ** 2

    def test_generator_uses_params_frame_size(self, base_config_values):
        from graphs.graph_generator import GraphGenerator

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
