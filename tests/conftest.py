import os
import pickle

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from graphs.synth_graph import SynthGraph  # noqa: E402

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


_SAMPLES_DIR = os.path.realpath(
    os.path.join(BASE_DIR, "..", "data", "input", "samples")
)


def _load_synth_graph(path: str) -> SynthGraph:
    if not os.path.exists(path):
        pytest.skip(f"Fixture data not found: {path}")
    with open(path, "rb") as f:
        graph = pickle.load(f)
    assert isinstance(graph, SynthGraph)
    return graph


def _has_sample_data() -> bool:
    """True when at least one dataset file is present under the samples tree.

    Checks for files, not just the directory: the pipelines create the input
    directory tree on startup, so an empty ``samples/`` is a normal state.
    """
    for _, _, filenames in os.walk(_SAMPLES_DIR):
        if any(f.endswith((".npy", ".pkl")) for f in filenames):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when the sample input data is absent.

    The code distribution ships without input datasets, so these tests would
    otherwise fail with a confusing "file does not exist" for anyone who has
    not yet downloaded the data (see REPRODUCING.md).
    """
    if _has_sample_data():
        return
    skip = pytest.mark.skip(
        reason=f"sample input data not found in {_SAMPLES_DIR} — see REPRODUCING.md"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


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
