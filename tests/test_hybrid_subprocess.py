# SPDX-License-Identifier: GPL-3.0-or-later
import pickle
from dataclasses import replace

import pytest

from networksynth.configs import DatasetId
from networksynth.handlers.run_logging import reset_logging
from networksynth.handlers.run_paths import RunPaths

pytestmark = pytest.mark.unit


@pytest.fixture
def config(tmp_path):
    from tests.fixture_config import FIXTURE_HYBRID

    return replace(FIXTURE_HYBRID, BASE_OUTPUT_PATH=str(tmp_path), RUN_ID="abc123")


@pytest.fixture(autouse=True)
def _no_leaked_handlers():
    yield
    reset_logging()


class TestDatasetIdCrossesProcesses:
    @pytest.mark.parametrize(
        "dataset_id", [DatasetId("a"), DatasetId("A", "10kX"), DatasetId("x", "y", "z")]
    )
    def test_it_survives_a_pickle_round_trip(self, dataset_id):
        restored = pickle.loads(pickle.dumps(dataset_id))

        assert restored == dataset_id
        assert tuple(restored) == tuple(dataset_id)
        assert restored.path == dataset_id.path


class TestTheChildSetsItselfUp:
    def test_it_runs_the_dataset_it_is_given(self, tmp_path, config, monkeypatch):
        import networksynth.pipelines.hybrid as hybrid

        ran = []
        monkeypatch.setattr(
            hybrid, "run_hybrid_for_dataset", lambda *a: ran.append(a[0])
        )

        hybrid._subprocess_target(DatasetId("ds"), config, RunPaths(root=str(tmp_path)))

        assert ran == [DatasetId("ds")]

    def test_it_logs_under_the_parents_run_id(self, tmp_path, config, monkeypatch):
        import networksynth.pipelines.hybrid as hybrid

        monkeypatch.setattr(hybrid, "run_hybrid_for_dataset", lambda *a: None)

        hybrid._subprocess_target(DatasetId("ds"), config, RunPaths(root=str(tmp_path)))

        assert config.RUN_ID == "abc123"
        assert (tmp_path / "run.jsonl").exists()


class TestAChildThatDiesFailsTheRun:
    def test_a_non_zero_exit_raises(self, tmp_path, config, monkeypatch):
        import networksynth.pipelines.hybrid as hybrid

        class Proc:
            exitcode = 1

            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        monkeypatch.setattr(
            hybrid, "spawn_context", lambda: type("Ctx", (), {"Process": Proc})
        )

        with pytest.raises(RuntimeError, match="exited with code 1"):
            hybrid._run_dataset_in_subprocess(
                DatasetId("ds"), config, RunPaths(root=str(tmp_path))
            )

    def test_a_clean_exit_does_not(self, tmp_path, config, monkeypatch):
        import networksynth.pipelines.hybrid as hybrid

        class Proc:
            exitcode = 0

            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        monkeypatch.setattr(
            hybrid, "spawn_context", lambda: type("Ctx", (), {"Process": Proc})
        )

        hybrid._run_dataset_in_subprocess(
            DatasetId("ds"), config, RunPaths(root=str(tmp_path))
        )
