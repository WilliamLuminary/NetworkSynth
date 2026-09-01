import os

import matplotlib

matplotlib.use("Agg")

import pytest

from graphs.graphml_io import read_graph_graphml
from graphs.synth_graph import SynthGraph

BASE_DIR = os.path.dirname(__file__)


_WEIGHTED_GRAPHML = os.path.realpath(
    os.path.join(BASE_DIR, "data", "A_10kX_weighted_network_positive.graphml")
)
_UNWEIGHTED_GRAPHML = os.path.realpath(
    os.path.join(BASE_DIR, "data", "sample1_unweighted_network.graphml")
)


def _load_synth_graph(path: str) -> SynthGraph:
    if not os.path.exists(path):
        pytest.skip(f"Fixture data not found: {path}")
    graph = read_graph_graphml(path)
    assert isinstance(graph, SynthGraph)
    return graph


@pytest.fixture
def load_weighted_test_synth_graph():
    return _load_synth_graph(_WEIGHTED_GRAPHML)


@pytest.fixture
def load_unweighted_test_synth_graph():
    return _load_synth_graph(_UNWEIGHTED_GRAPHML)
