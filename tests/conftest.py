import os
import pickle

import pytest

BASE_DIR = os.path.dirname(__file__)

try:
    from graphs.synth_graph import SynthGraph
except ImportError:
    SynthGraph = None

nx = pytest.importorskip("networkx", reason="Legacy pkl fixtures require networkx")


@pytest.fixture
def load_weighted_test_nx_graph():
    """Load raw nx.Graph (for backward-compat reference tests)."""
    file_path = os.path.realpath(
        os.path.join(BASE_DIR, "data", "A_10kX_weighted_network.pkl")
    )
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


@pytest.fixture
def load_weighted_test_synth_graph(load_weighted_test_nx_graph):
    """Weighted SynthGraph converted from the test pkl."""
    if SynthGraph is None:
        pytest.skip("SynthGraph requires networkit")
    return SynthGraph.from_networkx(load_weighted_test_nx_graph)


@pytest.fixture
def load_unweighted_test_nx_graph():
    """Load raw nx.Graph (for backward-compat reference tests)."""
    file_path = os.path.realpath(
        os.path.join(BASE_DIR, "data", "sample1_unweighted_network.pkl")
    )
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    return G


@pytest.fixture
def load_unweighted_test_synth_graph(load_unweighted_test_nx_graph):
    """Unweighted SynthGraph converted from the test pkl."""
    if SynthGraph is None:
        pytest.skip("SynthGraph requires networkit")
    return SynthGraph.from_networkx(load_unweighted_test_nx_graph)
