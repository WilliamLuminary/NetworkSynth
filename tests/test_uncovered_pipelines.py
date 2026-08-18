# tests/test_uncovered_pipelines.py
"""Smoke coverage for the two pipelines nothing else touched.

`generate_from_props` and `analyze` were the only pipelines with no test at all,
so a refactor could break either without anything noticing.

Kept deliberately tiny. Generation and multifractal analysis both scale badly
with node count, so every graph here is a few dozen nodes and every run asks for
one network with one attempt and the quality gate off. These prove the wiring
holds end to end; they are not performance or quality tests.
"""

import os
import pickle
from dataclasses import asdict

import networkit as nk
import numpy as np
import pytest

from configs import BaseConfig
from graphs.synth_graph import SynthGraph
from handlers.attributes_calculator import AttributesCalculator

pytestmark = pytest.mark.unit


def _small_graph(side=8, spacing=10.0, seed=3):
    """A small square lattice.

    Geometry matters more than size here: the generator grows a network using the
    edge-length distribution of its input, and requires more than 100 nodes to
    accept a result. Random positions give edges as long as the whole frame, so
    almost nothing fits and every attempt fails. A lattice gives short, uniform
    edges, which is both realistic and what makes a tiny test viable.
    """
    rng = np.random.default_rng(seed)
    coords = [
        (x * spacing, y * spacing) for y in range(side) for x in range(side)
    ]
    positions = np.asarray(coords, dtype=float)
    positions += rng.normal(0, spacing * 0.05, size=positions.shape)  # slight jitter

    graph = nk.Graph(len(coords), weighted=False)
    for y in range(side):
        for x in range(side):
            here = y * side + x
            if x + 1 < side:
                graph.addEdge(here, here + 1)
            if y + 1 < side:
                graph.addEdge(here, here + side)
    return SynthGraph(graph, positions)


# ---------------------------------------------------------------------------
# generate_from_props — generates from an attributes dict, with no source graph
# ---------------------------------------------------------------------------


@pytest.fixture
def attr_dir(tmp_path):
    """The directory shape the loader looks for: a *_property.pkl inside it."""
    attributes = AttributesCalculator().analyze(_small_graph())
    directory = tmp_path / "attrs"
    directory.mkdir()
    with open(directory / "original_property.pkl", "wb") as handle:
        pickle.dump(asdict(attributes), handle)
    return directory


class TestGenerateFromProps:
    def test_generates_a_network_from_an_attributes_dict(self, attr_dir, tmp_path):
        from configs.attr_generate_mode.config_sample import SampleConfig
        from pipelines.generate_from_props import run

        class TinyConfig(SampleConfig):
            ATTRIBUTES_DICT_DATA_PATH = str(attr_dir)
            BASE_OUTPUT_PATH = str(tmp_path / "out")
            # GraphGenerator hard-requires more than 100 nodes, so the frame
            # cannot shrink below roughly this without every attempt failing.
            FRAME_SIZE = (200, 200)
            SYNTHETIC_FRAME_SIZE = (200, 200)
            IMAGE_SIZE = (200, 200)
            SYNTHETIC_NETWORK_NUMBER = 1
            SYNTHETIC_GRAPH_NUMBER = 0
            MAX_ATTEMPTS = 5
            ERROR_CHECKER = "none"
            MEASURE_WEIGHTED = False
            SEED = 5

        TinyConfig.initialize()
        run(TinyConfig)

        # Attr mode hands the *input* directory to RunAgent as its output dir,
        # so results land beside the attributes rather than under BASE_OUTPUT_PATH.
        produced = list(attr_dir.rglob("*.csv"))
        assert produced, f"no output written; attr dir holds {list(attr_dir.rglob('*'))}"

    def test_the_loader_finds_the_property_pickle(self, attr_dir):
        """It scans for a name containing 'property' or 'attribute'."""
        from configs.attr_generate_mode.config_sample import SampleConfig

        loaded = SampleConfig._load_attr_dict(str(attr_dir))

        assert loaded, "attributes dict came back empty"
        assert "average_degree" in loaded


# ---------------------------------------------------------------------------
# analyze — discovers saved network directories and analyses what it finds
# ---------------------------------------------------------------------------


class TestAnalyzeDiscovery:
    """`find_pkl_containers` is the part most likely to break silently."""

    def test_finds_a_directory_holding_synthetic_and_original(self, tmp_path):
        from pipelines.analyze import find_pkl_containers

        container = tmp_path / "run_a" / "dataset"
        (container / "synthetic").mkdir(parents=True)
        (container / "original").mkdir(parents=True)

        found = find_pkl_containers(str(tmp_path))

        assert len(found) == 1
        assert os.path.realpath(list(found.values())[0]) == os.path.realpath(
            str(container)
        )

    def test_ignores_directories_without_network_subdirs(self, tmp_path):
        from pipelines.analyze import find_pkl_containers

        (tmp_path / "logs").mkdir()
        (tmp_path / "figures").mkdir()

        assert find_pkl_containers(str(tmp_path)) == {}

    def test_skips_dot_and_dunder_directories(self, tmp_path):
        from pipelines.analyze import find_pkl_containers

        hidden = tmp_path / ".cache" / "dataset"
        (hidden / "synthetic").mkdir(parents=True)

        assert find_pkl_containers(str(tmp_path)) == {}


class TestAnalyzeLoadsWhatItFinds:
    def test_run_agent_loads_the_discovered_networks(self, tmp_path):
        """The handoff from discovery to the batch processor.

        Stops short of `analyze_graph`, which is measured elsewhere and too slow
        to belong in a smoke test.
        """
        from configs.analyze_mode.config_sample import SampleConfig
        from handlers import RunAgent

        container = tmp_path / "dataset"
        for kind in ("synthetic", "original"):
            (container / kind).mkdir(parents=True)
        with open(container / "synthetic" / "synthetic_network.pkl", "wb") as handle:
            pickle.dump([_small_graph(), _small_graph(seed=4)], handle)
        with open(container / "original" / "original_network.pkl", "wb") as handle:
            pickle.dump([_small_graph(seed=5)], handle)

        class TinyConfig(SampleConfig):
            BASE_OUTPUT_PATH = str(tmp_path / "out")
            MEASURE_WEIGHTED = False
            FULL_Q_BAND = False

        TinyConfig.initialize()
        agent = RunAgent(TinyConfig, networks_path=str(container))
        agent.prepare_data()

        assert agent.batch_processor is not None
        assert len(agent.data_loader.get_synthetic_networks()) == 2


def test_base_config_is_untouched_by_these_runs():
    """The configs above are subclasses; their values must not leak upward.

    This is what `_inject_dependencies` used to break: one config's settings
    landed on BaseConfig and the next run inherited them.
    """
    assert BaseConfig.DISABLE_SAVING is False
    assert BaseConfig.ERROR_CHECKER == "multifractal", "a subclass overwrote it"
    assert BaseConfig.SYNTHETIC_NETWORK_NUMBER != 1, "a subclass overwrote it"
