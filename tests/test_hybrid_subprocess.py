import pickle

import pytest

from configs import BaseConfig, DatasetId
from handlers.run_logging import reset_logging
from handlers.run_paths import RunPaths

pytestmark = pytest.mark.unit


@pytest.fixture
def config(tmp_path):
    class Config(BaseConfig):
        BASE_OUTPUT_PATH = str(tmp_path)
        DATASETS = [DatasetId("ds")]
        initialised = False

        @classmethod
        def initialize(cls):
            super().initialize()
            cls.initialised = True

    return Config


@pytest.fixture(autouse=True)
def _no_leaked_handlers():
    yield
    reset_logging()


class TestDatasetIdCrossesProcesses:
    @pytest.mark.parametrize(
        "dataset_id", [DatasetId("a"), DatasetId("A", "10kX"), DatasetId("x", "y", "z")]
    )
    def test_it_survives_a_pickle_round_trip(self, dataset_id):
        """A variadic constructor must not cost picklability.

        Rebuilding a tuple subclass hands the whole tuple to ``__new__`` as one
        argument, so without ``__getnewargs__`` the levels come back nested and
        a DatasetId sent to a spawned child never arrives intact.
        """
        restored = pickle.loads(pickle.dumps(dataset_id))

        assert restored == dataset_id
        assert tuple(restored) == tuple(dataset_id)
        assert restored.path == dataset_id.path


class TestTheChildSetsItselfUp:
    def test_it_initializes_the_config_it_receives(self, tmp_path, config, monkeypatch):
        """Nothing the parent did to the config class travels to the child."""
        import pipelines.hybrid as hybrid

        ran = []
        monkeypatch.setattr(
            hybrid, "run_hybrid_for_dataset", lambda *a: ran.append(a[0])
        )

        hybrid._subprocess_target(
            DatasetId("ds"), config, RunPaths(root=str(tmp_path)), "abc123"
        )

        assert config.initialised, "the pipeline ran with an uninitialised config"
        assert ran, "the pipeline never ran"

    def test_it_logs_under_the_parents_run_id(self, tmp_path, config, monkeypatch):
        import pipelines.hybrid as hybrid

        monkeypatch.setattr(hybrid, "run_hybrid_for_dataset", lambda *a: None)

        hybrid._subprocess_target(
            DatasetId("ds"), config, RunPaths(root=str(tmp_path)), "abc123"
        )

        # Same id and same file as the parent, or the GUI tailing run.jsonl sees
        # nothing at all while a dataset is being built.
        assert config.RUN_ID == "abc123"
        assert (tmp_path / "run.jsonl").exists()


class TestAChildThatDiesFailsTheRun:
    def test_a_non_zero_exit_raises(self, tmp_path, config, monkeypatch):
        import pipelines.hybrid as hybrid

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
        import pipelines.hybrid as hybrid

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
