import os
import pickle

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

try:
    from graphs.synth_graph import SynthGraph  # noqa: E402
except ImportError:
    SynthGraph = None

BASE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Fixture-data paths
# ---------------------------------------------------------------------------

_WEIGHTED_PKL = os.path.realpath(
    os.path.join(BASE_DIR, "data", "A_10kX_weighted_network.pkl")
)
_UNWEIGHTED_PKL = os.path.realpath(
    os.path.join(BASE_DIR, "data", "sample1_unweighted_network.pkl")
)


def _load_synth_graph(path: str):
    if SynthGraph is None:
        pytest.skip("SynthGraph requires networkit")
    if not os.path.exists(path):
        pytest.skip(f"Fixture data not found: {path}")
    with open(path, "rb") as f:
        graph = pickle.load(f)
    assert isinstance(graph, SynthGraph)
    return graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def load_weighted_test_synth_graph():
    """Weighted SynthGraph fixture."""
    return _load_synth_graph(_WEIGHTED_PKL)


@pytest.fixture
def load_unweighted_test_synth_graph():
    """Unweighted SynthGraph fixture."""
    return _load_synth_graph(_UNWEIGHTED_PKL)
