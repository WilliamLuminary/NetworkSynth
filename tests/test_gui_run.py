# tests/test_gui_run.py
"""
The GUI entry path: a JSON run-spec in, a generated network plus manifest out.

This is what StructuralGT's controller drives. It never imports our code — it
writes a spec, launches ``gui_run.py``, reads the exit code, tails ``run.jsonl``
and reads ``manifest.json``. These tests exercise that same surface.
"""

import json
import os

import pytest

import gui_run
from configs.gui_config import GuiConfig, SpecError

pytestmark = pytest.mark.unit

#: Derived, never restated: a mode list written out by hand goes stale the moment
#: one is wired or unwired, and then the tests quietly stop covering it.
_MODE_NAMES = set(gui_run._MODES)


def _spec(tmp_path, **overrides):
    spec = {
        "contract": 1,
        "mode": "generate",
        "run_name": "test_run",
        "output_dir": str(tmp_path / "out"),
        "inputs": {
            "edge_list": str(tmp_path / "e.csv"),
            "positions": str(tmp_path / "p.csv"),
            "image": None,
        },
        "params": {"FRAME_SIZE": [512, 512], "SEED": 7},
    }
    spec.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return str(path)


class TestSpecLoading:
    def test_builds_a_config_from_a_spec(self, tmp_path):
        config = GuiConfig.from_spec(_spec(tmp_path))

        assert config.MODE == "generate"
        assert config.BASE_OUTPUT_PATH == str(tmp_path / "out")
        assert [str(d) for d in config.DATASETS] == ["test_run"]

    def test_json_arrays_become_tuples(self, tmp_path):
        """Pipelines unpack these as tuples; JSON only has arrays."""
        config = GuiConfig.from_spec(_spec(tmp_path))

        assert config.FRAME_SIZE == (512, 512)
        assert isinstance(config.FRAME_SIZE, tuple)

    def test_scalar_params_pass_through(self, tmp_path):
        config = GuiConfig.from_spec(_spec(tmp_path))

        assert config.SEED == 7

    def test_returns_a_fresh_subclass_each_time(self, tmp_path):
        """Two specs in one process must not collide."""
        first = GuiConfig.from_spec(_spec(tmp_path, run_name="one"))
        second_dir = tmp_path / "second"
        second_dir.mkdir()
        second = GuiConfig.from_spec(_spec(second_dir, run_name="two"))

        assert first is not second
        assert first is not GuiConfig
        assert [str(d) for d in first.DATASETS] == ["one"]
        assert [str(d) for d in second.DATASETS] == ["two"]

    def test_output_denote_names_the_mode(self, tmp_path):
        config = GuiConfig.from_spec(_spec(tmp_path))
        config.initialize()

        assert config.OUTPUT_DENOTE == "gui_generate"


class TestSpecIsValidated:
    """A bad spec must be rejected clearly, not half-understood."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(SpecError, match="not found"):
            GuiConfig.from_spec(str(tmp_path / "nope.json"))

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "spec.json"
        path.write_text("{not json")
        with pytest.raises(SpecError, match="not valid JSON"):
            GuiConfig.from_spec(str(path))

    @pytest.mark.parametrize("key", ["mode", "output_dir", "inputs", "params"])
    def test_missing_required_key(self, tmp_path, key):
        spec = json.loads(open(_spec(tmp_path)).read())
        del spec[key]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(spec))

        with pytest.raises(SpecError, match="missing required key"):
            GuiConfig.from_spec(str(path))

    def test_wrong_contract_version(self, tmp_path):
        with pytest.raises(SpecError, match="contract version"):
            GuiConfig.from_spec(_spec(tmp_path, contract=99))

    @pytest.mark.parametrize("key", ["edge_list", "positions"])
    def test_missing_input_path(self, tmp_path, key):
        spec = json.loads(open(_spec(tmp_path)).read())
        spec["inputs"][key] = None
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(spec))

        with pytest.raises(SpecError, match=f"inputs.{key} is required"):
            GuiConfig.from_spec(str(path))


class TestEntryPointExitCodes:
    """The caller reads these to tell success from cancel from breakage."""

    def test_bad_spec_exits_2(self, tmp_path):
        assert gui_run.main(["gui_run.py", str(tmp_path / "nope.json")]) == 2

    def test_no_spec_argument_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NETWORKSYNTH_RUN_SPEC", raising=False)

        assert gui_run.main(["gui_run.py"]) == 2

    def test_unsupported_mode_exits_2(self, tmp_path):
        """Better a clear rejection than a confusing failure mid-run.

        Uses a name that will never be a mode. This once used "hybrid", which
        then became supported and turned the test into a false alarm.
        """
        assert gui_run.main(["gui_run.py", _spec(tmp_path, mode="not_a_mode")]) == 2

    def test_reads_the_spec_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NETWORKSYNTH_RUN_SPEC", _spec(tmp_path, mode="nope"))

        # mode is rejected, which proves the env var was read at all
        assert gui_run.main(["gui_run.py"]) == 2

    def test_pipeline_failure_exits_1(self, tmp_path, monkeypatch):
        import pipelines.generate as gen

        monkeypatch.setattr(
            gen, "main", lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 1

    def test_cancellation_exits_130(self, tmp_path, monkeypatch):
        import pipelines.generate as gen

        monkeypatch.setattr(
            gen, "main", lambda **k: (_ for _ in ()).throw(KeyboardInterrupt)
        )

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 130

    def test_success_exits_0(self, tmp_path, monkeypatch):
        import pipelines.generate as gen

        monkeypatch.setattr(gen, "main", lambda **k: None)

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 0


@pytest.mark.integration
@pytest.mark.requires_fixture_data
class TestFullRun:
    def test_spec_in_network_and_manifest_out(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        """The whole contract: their CSVs in, our outputs discoverable."""
        from configs.file_definitions import save_network_csv

        save_network_csv(load_unweighted_test_synth_graph, str(tmp_path / "net.csv"))

        spec = {
            "contract": 1,
            "mode": "generate",
            "run_name": "sample",
            "output_dir": str(tmp_path / "out"),
            "inputs": {
                "edge_list": str(tmp_path / "net_edgelist.csv"),
                "positions": str(tmp_path / "net_positions.csv"),
                "image": None,
            },
            "params": {
                "FRAME_SIZE": [512, 512],
                "SYNTHETIC_FRAME_SIZE": [512, 512],
                "IMAGE_SIZE": [512, 512],
                "CLOSED_NODES_FACTOR": 1.2,
                "CLOSED_EDGES_FACTOR": 0.8,
                "SYNTHETIC_NETWORK_NUMBER": 1,
                "SYNTHETIC_GRAPH_NUMBER": 0,
                "MAX_ATTEMPTS": 2,
                "ERROR_CHECKER": "none",
                "MEASURE_WEIGHTED": False,
                "SEED": 11,
            },
        }
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        assert gui_run.main(["gui_run.py", str(spec_path)]) == 0

        out = tmp_path / "out"
        roots = [d for d in os.listdir(out) if d.startswith("gui_generate_results_")]
        assert len(roots) == 1, f"expected one run dir, got {roots}"
        root = out / roots[0]

        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "ok"
        assert manifest["outputs"].get("edge_lists"), manifest["outputs"]

        assert (root / "run.jsonl").exists(), "log must be inside the output dir"


class TestModeSelection:
    """The GUI offers a mode per pipeline, so the two tables must agree.

    A mode offered in the window but absent from the dispatch table fails at
    click time; the reverse is a pipeline nobody can reach.
    """

    def test_offered_modes_match_dispatchable_modes(self):
        from gui.spec_builder import MODES

        assert set(MODES) == set(gui_run._MODES)

    @pytest.mark.parametrize("mode", sorted(_MODE_NAMES))
    def test_every_mode_builds_a_valid_spec(self, mode, tmp_path):
        from gui import spec_builder

        edge_list = tmp_path / "e.csv"
        positions = tmp_path / "p.csv"
        edge_list.write_text("source_index,target_index\n0,1\n")
        positions.write_text("x,y\n0,0\n1,1\n")

        values = spec_builder.default_values(mode)
        problems = spec_builder.validate(
            mode, str(edge_list), str(positions), str(tmp_path), values
        )
        assert not problems, problems

        spec = spec_builder.build_spec(
            mode, str(edge_list), str(positions), str(tmp_path), values
        )
        assert spec["mode"] == mode
        assert json.loads(json.dumps(spec)), "spec must survive JSON"

    #: Attributes each mode's pipeline reads that BaseConfig does not define.
    _MODE_EXTRAS = {
        "scaling": [
            "SCALE_ROWS",
            "SCALE_COLS",
            "ROOT_SPACING_FACTOR",
            "MAX_GENERATION_ROUNDS",
        ],
        "hybrid": ["TARGET_SCALE", "PHASE2_MAX_ROUNDS"],
    }

    @pytest.mark.parametrize("mode", sorted(_MODE_EXTRAS))
    def test_mode_specific_params_reach_the_config(self, mode, tmp_path):
        """These are read as plain attributes, so a missing one crashes mid-run."""
        from gui import spec_builder

        if mode not in _MODE_NAMES:
            pytest.skip(f"{mode} is not currently offered by the GUI")
        required = self._MODE_EXTRAS[mode]

        values = spec_builder.default_values(mode)
        spec = spec_builder.build_spec(mode, "e.csv", "p.csv", str(tmp_path), values)
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        config = GuiConfig.from_spec(str(spec_path))

        for name in required:
            assert hasattr(config, name), f"{mode} needs {name}"

    def test_size_params_become_tuples(self, tmp_path):
        """Pipelines unpack these; JSON only has arrays."""
        from gui import spec_builder

        values = spec_builder.default_values("hybrid")
        spec = spec_builder.build_spec(
            "hybrid", "e.csv", "p.csv", str(tmp_path), values
        )
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        config = GuiConfig.from_spec(str(spec_path))

        assert isinstance(config.TARGET_SCALE, tuple)
        assert isinstance(config.SYNTHETIC_FRAME_SIZE, tuple)


class TestSpecConfigCrossesProcesses:
    """A spec-built config must survive being pickled.

    ``from_spec`` returns a class created at run time, and pickle stores classes
    by name — a spawned child re-imports ``configs.gui_config`` and finds no such
    name. Under ``fork`` this never showed, because the child inherited the class
    object in memory. Under ``spawn`` ``hybrid`` failed outright: it hands the
    config class to a subprocess, where every other mode passes a ``SynthParams``.
    """

    def test_a_spec_built_config_survives_pickling(self, tmp_path):
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert restored.MODE == config.MODE
        assert restored.SEED == config.SEED
        assert restored.BASE_OUTPUT_PATH == config.BASE_OUTPUT_PATH
        assert [str(d) for d in restored.DATASETS] == [str(d) for d in config.DATASETS]

    def test_tuple_params_stay_tuples_after_a_round_trip(self, tmp_path):
        """Pipelines unpack these; a list would fail deep inside a worker."""
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert isinstance(restored.FRAME_SIZE, tuple)
        assert restored.FRAME_SIZE == config.FRAME_SIZE

    def test_the_base_class_still_pickles_by_name(self, tmp_path):
        """Only spec-built subclasses need rebuilding."""
        import pickle

        assert pickle.loads(pickle.dumps(GuiConfig)) is GuiConfig

    def test_the_loader_paths_survive(self, tmp_path):
        """A worker that cannot find its inputs is the failure this prevents."""
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert restored.PATHS["edge_list"] == config.PATHS["edge_list"]
        assert restored.PATHS["positions"] == config.PATHS["positions"]
