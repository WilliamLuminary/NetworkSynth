# SPDX-License-Identifier: GPL-3.0-or-later
import igraph as ig
import numpy as np
import pytest

from networksynth.configs import BaseConfig, DatasetId
from networksynth.graphs.synth_graph import SynthGraph

pytestmark = pytest.mark.unit


def _small_graph(side=8, spacing=10.0, seed=3):
    rng = np.random.default_rng(seed)
    coords = [(x * spacing, y * spacing) for y in range(side) for x in range(side)]
    positions = np.asarray(coords, dtype=float)
    positions += rng.normal(0, spacing * 0.05, size=positions.shape)

    pairs = []
    for y in range(side):
        for x in range(side):
            here = y * side + x
            if x + 1 < side:
                pairs.append((here, here + 1))
            if y + 1 < side:
                pairs.append((here, here + side))
    return SynthGraph(ig.Graph(n=len(coords), edges=pairs), positions)


class TestGenerationDoesNotAnalyse:
    def test_generation_has_no_analysis_of_its_own(self):
        import inspect

        import networksynth.pipelines.generate

        source = inspect.getsource(networksynth.pipelines.generate)
        for forbidden in (
            "MultifractalBatchProcessor",
            "analysis_data",
            "analysis_figure",
        ):
            assert forbidden not in source, f"generate reaches for {forbidden}"

    def test_the_generation_run_has_no_analysis_methods(self):
        from networksynth.handlers import GenerationRun

        for forbidden in ("multifractal_analysis", "analyse", "save_analysis"):
            assert not hasattr(GenerationRun, forbidden), forbidden


class TestAnalyzeReadsEveryFormatWeWrite:
    def test_reads_a_csv_pair_when_nothing_is_pickled(self, tmp_path):
        from networksynth.configs.compare_mode.config_sample import _load_networks
        from networksynth.configs.file_definitions import save_network_csv

        folder = tmp_path / "original"
        folder.mkdir()
        save_network_csv(_small_graph(), str(folder / "original_network.csv"))

        loaded = _load_networks(str(folder))

        assert len(loaded) == 1
        assert loaded[0].number_of_nodes() == 64

    def test_graphml_wins_over_the_csv_pair(self, tmp_path):
        from networksynth.configs.compare_mode.config_sample import _load_networks
        from networksynth.configs.file_definitions import save_network_csv
        from networksynth.graphs.graphml_io import write_graph_graphml

        folder = tmp_path / "synthetic"
        folder.mkdir()
        save_network_csv(_small_graph(), str(folder / "synthetic_network.csv"))
        for index in range(2):
            write_graph_graphml(
                _small_graph(seed=index),
                str(folder / f"synthetic_network_n{index}.graphml"),
            )

        assert len(_load_networks(str(folder))) == 2

    def test_an_edge_list_without_its_positions_stops_the_run(self, tmp_path):
        from networksynth.configs.compare_mode.config_sample import _load_networks

        folder = tmp_path / "original"
        folder.mkdir()
        (folder / "orphan_edgelist.csv").write_text("source_index,target_index\n0,1\n")

        with pytest.raises(AssertionError, match="no positions file"):
            _load_networks(str(folder))


class TestComparisonLoadsTheTwoSetsItIsGiven:
    def _config(self, tmp_path, original, synthetic):
        from tests.fixture_config import FixtureCompareConfig

        class TinyConfig(FixtureCompareConfig):
            BASE_OUTPUT_PATH = str(tmp_path / "out")
            MEASURE_WEIGHTED = False
            FULL_Q_BAND = False
            ORIGINAL_NETWORKS_PATH = str(original)
            SYNTHETIC_NETWORKS_PATH = str(synthetic)

        TinyConfig.initialize()
        return TinyConfig

    def _run(self, config):
        from networksynth.handlers import ComparisonRun, create_run_paths

        return ComparisonRun(
            config,
            create_run_paths(config),
            DatasetId("analysis"),
            config.ORIGINAL_NETWORKS_PATH,
            config.SYNTHETIC_NETWORKS_PATH,
        )

    def test_it_loads_both_sides(self, tmp_path):
        original, synthetic = tmp_path / "orig", tmp_path / "synth"
        original.mkdir()
        synthetic.mkdir()
        from networksynth.graphs.graphml_io import write_graph_graphml

        write_graph_graphml(
            _small_graph(seed=5), str(original / "original_network.graphml")
        )
        for index, seed in enumerate((3, 4)):
            write_graph_graphml(
                _small_graph(seed=seed),
                str(synthetic / f"synthetic_network_n{index}.graphml"),
            )

        run = self._run(self._config(tmp_path, original, synthetic))

        assert len(run.synthetic) == 2
        assert len(run.original) == 1

    def test_the_two_sets_can_be_any_directories(self, tmp_path):
        from networksynth.configs.file_definitions import save_network_csv

        left, right = tmp_path / "monday", tmp_path / "tuesday"
        left.mkdir()
        right.mkdir()
        save_network_csv(_small_graph(), str(left / "a.csv"))
        save_network_csv(_small_graph(seed=4), str(right / "b.csv"))

        run = self._run(self._config(tmp_path, left, right))

        assert len(run.original) == 1
        assert len(run.synthetic) == 1

    def test_it_writes_its_analysis_where_the_run_paths_say(self, tmp_path):
        from networksynth.configs.file_definitions import save_network_csv

        left, right = tmp_path / "a", tmp_path / "b"
        left.mkdir()
        right.mkdir()
        save_network_csv(_small_graph(), str(left / "a.csv"))
        save_network_csv(_small_graph(seed=4), str(right / "b.csv"))

        run = self._run(self._config(tmp_path, left, right))
        run.analyse()
        run.save_analysis()

        written = list((tmp_path / "out").rglob("*"))
        assert any(f.name.endswith("analysis_data.json") for f in written), written
        assert any("analysis_figure" in f.name for f in written), written


def test_base_config_is_untouched_by_these_runs():
    assert BaseConfig.DISABLE_SAVING is False
    assert BaseConfig.ERROR_CHECKER == "multifractal", "a subclass overwrote it"
    assert BaseConfig.SYNTHETIC_NETWORK_NUMBER != 1, "a subclass overwrote it"
