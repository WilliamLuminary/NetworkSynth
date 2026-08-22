import os
import pickle

import networkit as nk
import numpy as np
import pytest

from configs import BaseConfig, DatasetId
from graphs.synth_graph import SynthGraph

pytestmark = pytest.mark.unit


def _small_graph(side=8, spacing=10.0, seed=3):
    rng = np.random.default_rng(seed)
    coords = [(x * spacing, y * spacing) for y in range(side) for x in range(side)]
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
# analyze — discovers saved network directories and analyses what it finds
# ---------------------------------------------------------------------------


class TestAnalyzeDiscovery:

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


class TestAnalyzeReadsEveryFormatWeWrite:
    def test_reads_a_csv_pair_when_nothing_is_pickled(self, tmp_path):
        from configs.analyze_mode.config_sample import _load_networks
        from configs.file_definitions import save_network_csv

        folder = tmp_path / "original"
        folder.mkdir()
        save_network_csv(_small_graph(), str(folder / "original_network.csv"))

        loaded = _load_networks(str(folder))

        assert len(loaded) == 1
        assert loaded[0].number_of_nodes() == 64

    def test_a_pickle_wins_over_the_csv_pair(self, tmp_path):
        from configs.analyze_mode.config_sample import _load_networks
        from configs.file_definitions import save_network_csv

        folder = tmp_path / "synthetic"
        folder.mkdir()
        save_network_csv(_small_graph(), str(folder / "synthetic_network.csv"))
        with open(folder / "synthetic_network.pkl", "wb") as handle:
            pickle.dump([_small_graph(), _small_graph(seed=9)], handle)

        # Both formats hold the same run; counting them twice would inflate
        # every statistic the analysis reports.
        assert len(_load_networks(str(folder))) == 2

    def test_an_edge_list_without_its_positions_stops_the_run(self, tmp_path):
        from configs.analyze_mode.config_sample import _load_networks

        folder = tmp_path / "original"
        folder.mkdir()
        (folder / "orphan_edgelist.csv").write_text("source_index,target_index\n0,1\n")

        with pytest.raises(AssertionError, match="no positions file"):
            _load_networks(str(folder))


class TestAnalyzeLoadsWhatItFinds:
    def test_run_agent_loads_the_discovered_networks(self, tmp_path):
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
        from handlers import create_run_paths

        agent = RunAgent(
            TinyConfig,
            networks_path=str(container),
            run_paths=create_run_paths(TinyConfig),
            dataset_id=DatasetId("dataset"),
        )
        agent.prepare_data()

        assert agent.batch_processor is not None
        assert len(agent.data_loader.get_synthetic_networks()) == 2


def test_base_config_is_untouched_by_these_runs():
    assert BaseConfig.DISABLE_SAVING is False
    assert BaseConfig.ERROR_CHECKER == "multifractal", "a subclass overwrote it"
    assert BaseConfig.SYNTHETIC_NETWORK_NUMBER != 1, "a subclass overwrote it"
