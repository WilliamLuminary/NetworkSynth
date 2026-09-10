# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

import pytest

from networksynth.configs import CompareConfig, DatasetId
from networksynth.handlers import NullSaver, Saver, build_saver
from networksynth.handlers.run_paths import RunPaths

pytestmark = pytest.mark.unit


def _config(tmp_path, disabled):
    from tests.fixture_config import FIXTURE

    return replace(
        FIXTURE,
        BASE_OUTPUT_PATH=str(tmp_path),
        DATASETS=[DatasetId("ds")],
        DISABLE_SAVING=disabled,
        DISABLE_SAVING_NOTE="a test",
    )


class _NoNetworks(CompareConfig):
    def load_networks(self, path):
        return []


class TestBuildSaver:
    def test_saving_on_gives_a_real_saver(self, tmp_path):
        saver = build_saver(_config(tmp_path, False), str(tmp_path / "out"))

        assert type(saver) is Saver
        assert (tmp_path / "out").is_dir()

    def test_saving_off_gives_a_null_saver(self, tmp_path):
        saver = build_saver(_config(tmp_path, True), str(tmp_path / "out"))

        assert isinstance(saver, NullSaver)
        assert not (tmp_path / "out").exists(), "a disabled run left a trace"


class TestNullSaverIsSaferThanNone:

    @pytest.fixture
    def saver(self, tmp_path):
        return build_saver(_config(tmp_path, True), str(tmp_path / "out"))

    def test_it_accepts_a_save_and_writes_nothing(self, saver, tmp_path):
        saver.save({"anything": 1}, "analysis_data")

        assert not (tmp_path / "out").exists()

    def test_it_accepts_a_batch(self, saver):
        saver.begin_batch()
        saver.save({"a": 1}, "analysis_data")
        saver.end_batch()

    def test_it_still_reports_where_it_would_have_written(self, saver, tmp_path):
        assert saver.output_dir == str(tmp_path / "out")

    def test_it_is_truthy_so_nothing_treats_it_as_missing(self, saver):
        assert saver

    def test_a_disabled_generation_run_can_be_saved_through(self, tmp_path):
        from networksynth.handlers import GenerationRun, create_run_paths

        config = _config(tmp_path, True)
        run = GenerationRun(config, create_run_paths(config), DatasetId("ds"))

        run.saver.begin_batch()
        run.save({"a": 1}, "analysis_data")
        run.saver.end_batch()

        assert list(tmp_path.iterdir()) == [], "a disabled run wrote something"

    def test_a_comparison_run_can_be_disabled(self, tmp_path):
        from networksynth.handlers import ComparisonRun

        original, synthetic = tmp_path / "orig", tmp_path / "synth"
        original.mkdir()
        synthetic.mkdir()

        config = _NoNetworks(
            DATASETS=[DatasetId("ds")],
            BASE_OUTPUT_PATH=str(tmp_path),
            DISABLE_SAVING=True,
            MEASURE_WEIGHTED=False,
            ORIGINAL_NETWORKS_PATH="",
            SYNTHETIC_NETWORKS_PATH="",
        )

        run = ComparisonRun(
            config,
            RunPaths(root=str(tmp_path / "run")),
            DatasetId("ds"),
            str(original),
            str(synthetic),
        )

        assert isinstance(run.saver, NullSaver)
        assert not (tmp_path / "run").exists()
