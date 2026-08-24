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


class TestGenerationDoesNotAnalyse:
    def test_a_generation_run_refuses_to_analyse(self, tmp_path):
        """The two are separate modes: generation makes networks, nothing else."""
        from configs import BaseConfig, DatasetId
        from handlers import RunAgent, create_run_paths

        class TinyConfig(BaseConfig):
            BASE_OUTPUT_PATH = str(tmp_path / "out")
            DATASETS = [DatasetId("ds")]

        TinyConfig.initialize()
        agent = RunAgent(
            TinyConfig,
            run_paths=create_run_paths(TinyConfig),
            dataset_id=DatasetId("ds"),
        )

        with pytest.raises(AssertionError, match="analyse mode's job"):
            agent.multifractal_analysis()


class TestAnalyzeReadsEveryFormatWeWrite:
    def test_reads_a_csv_pair_when_nothing_is_pickled(self, tmp_path):
        from configs.compare_mode.config_sample import _load_networks
        from configs.file_definitions import save_network_csv

        folder = tmp_path / "original"
        folder.mkdir()
        save_network_csv(_small_graph(), str(folder / "original_network.csv"))

        loaded = _load_networks(str(folder))

        assert len(loaded) == 1
        assert loaded[0].number_of_nodes() == 64

    def test_a_pickle_wins_over_the_csv_pair(self, tmp_path):
        from configs.compare_mode.config_sample import _load_networks
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
        from configs.compare_mode.config_sample import _load_networks

        folder = tmp_path / "original"
        folder.mkdir()
        (folder / "orphan_edgelist.csv").write_text("source_index,target_index\n0,1\n")

        with pytest.raises(AssertionError, match="no positions file"):
            _load_networks(str(folder))


class TestAnalyzeLoadsTheTwoSetsItIsGiven:
    def _config(self, tmp_path, original, synthetic):
        from configs.compare_mode.config_sample import SampleConfig

        class TinyConfig(SampleConfig):
            BASE_OUTPUT_PATH = str(tmp_path / "out")
            MEASURE_WEIGHTED = False
            FULL_Q_BAND = False
            ORIGINAL_NETWORKS_PATH = str(original)
            SYNTHETIC_NETWORKS_PATH = str(synthetic)

        TinyConfig.initialize()
        return TinyConfig

    def _agent(self, config, tmp_path):
        from handlers import RunAgent, create_run_paths

        return RunAgent(
            config,
            run_paths=create_run_paths(config),
            dataset_id=DatasetId("analysis"),
            original_path=config.ORIGINAL_NETWORKS_PATH,
            synthetic_path=config.SYNTHETIC_NETWORKS_PATH,
        )

    def test_it_loads_both_sides(self, tmp_path):
        original, synthetic = tmp_path / "orig", tmp_path / "synth"
        original.mkdir()
        synthetic.mkdir()
        with open(original / "original_network.pkl", "wb") as handle:
            pickle.dump([_small_graph(seed=5)], handle)
        with open(synthetic / "synthetic_network.pkl", "wb") as handle:
            pickle.dump([_small_graph(), _small_graph(seed=4)], handle)

        agent = self._agent(self._config(tmp_path, original, synthetic), tmp_path)
        agent.prepare_data()

        assert agent.batch_processor is not None
        assert len(agent.data_loader.get_synthetic_networks()) == 2
        assert len(agent.data_loader.get_original_network()) == 1

    def test_the_two_sets_can_be_any_directories(self, tmp_path):
        """Nothing about a results layout is assumed — the caller chooses."""
        from configs.file_definitions import save_network_csv

        left, right = tmp_path / "monday", tmp_path / "tuesday"
        left.mkdir()
        right.mkdir()
        save_network_csv(_small_graph(), str(left / "a.csv"))
        save_network_csv(_small_graph(seed=4), str(right / "b.csv"))

        agent = self._agent(self._config(tmp_path, left, right), tmp_path)
        agent.prepare_data()

        assert len(agent.data_loader.get_original_network()) == 1
        assert len(agent.data_loader.get_synthetic_networks()) == 1

    def test_one_set_alone_is_refused(self, tmp_path):
        from handlers import RunAgent, create_run_paths

        config = self._config(tmp_path, tmp_path, tmp_path)
        with pytest.raises(AssertionError, match="analysis compares two sets"):
            RunAgent(
                config,
                run_paths=create_run_paths(config),
                dataset_id=DatasetId("analysis"),
                original_path=str(tmp_path),
            )


def test_base_config_is_untouched_by_these_runs():
    assert BaseConfig.DISABLE_SAVING is False
    assert BaseConfig.ERROR_CHECKER == "multifractal", "a subclass overwrote it"
    assert BaseConfig.SYNTHETIC_NETWORK_NUMBER != 1, "a subclass overwrote it"
