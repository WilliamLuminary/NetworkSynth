# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os

import pytest

from networksynth import gui_run
from networksynth.configs.gui_config import GuiConfig, SpecError

pytestmark = pytest.mark.unit

_MODE_NAMES = set(gui_run._GUI_MODES)


def _inputs_for(mode, tmp_path, shape_index=0):
    from networksynth.gui import spec_builder

    shape = spec_builder.MODES[mode].input_shapes[shape_index]
    made = {}
    for spec_input in shape.inputs:
        target = tmp_path / spec_input.id
        if spec_input.kind == "dir":
            target.mkdir(exist_ok=True)
            if spec_input.id == "datasets_dir":
                (target / "one_edgelist.csv").write_text(
                    "source_index,target_index\n0,1\n"
                )
                (target / "one_positions.csv").write_text("x,y\n0,0\n1,1\n")
        else:
            target.write_text("x\n")
        made[spec_input.id] = str(target)
    return made


def _every_shape():
    from networksynth.gui import spec_builder

    return [
        (mode, index)
        for mode in sorted(_MODE_NAMES)
        for index in range(len(spec_builder.MODES[mode].input_shapes))
    ]


def _spec(tmp_path, **overrides):
    spec = {
        "contract": 2,
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
        config = GuiConfig.from_spec(_spec(tmp_path))

        assert config.FRAME_SIZE == (512, 512)
        assert isinstance(config.FRAME_SIZE, tuple)

    def test_scalar_params_pass_through(self, tmp_path):
        config = GuiConfig.from_spec(_spec(tmp_path))

        assert config.SEED == 7

    def test_returns_a_fresh_subclass_each_time(self, tmp_path):
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

    def test_unknown_mode_is_rejected(self, tmp_path):
        with pytest.raises(SpecError, match="unknown mode"):
            GuiConfig.from_spec(_spec(tmp_path, mode="not_a_mode"))

    def test_keys_from_another_shape_are_not_an_input_set(self, tmp_path):
        spec = json.loads(open(_spec(tmp_path)).read())
        spec["inputs"] = {"original_dir": str(tmp_path), "synthetic_dir": str(tmp_path)}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(spec))

        with pytest.raises(SpecError, match="fills none of its input sets"):
            GuiConfig.from_spec(str(path))

    def test_filling_two_input_sets_is_rejected(self, tmp_path):
        datasets = tmp_path / "many"
        datasets.mkdir()
        spec = json.loads(open(_spec(tmp_path)).read())
        spec["inputs"]["datasets_dir"] = str(datasets)
        path = tmp_path / "both.json"
        path.write_text(json.dumps(spec))

        with pytest.raises(SpecError, match="matches more than one"):
            GuiConfig.from_spec(str(path))

    @pytest.mark.parametrize("key", ["edge_list", "positions"])
    def test_half_a_pair_is_not_an_input_set(self, tmp_path, key):
        spec = json.loads(open(_spec(tmp_path)).read())
        spec["inputs"][key] = None
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(spec))

        with pytest.raises(SpecError, match="fills none of its input sets"):
            GuiConfig.from_spec(str(path))


class TestDirectoryInput:

    def _pair(self, directory, name):
        (directory / f"{name}_edgelist.csv").write_text(
            "source_index,target_index\n0,1\n"
        )
        (directory / f"{name}_positions.csv").write_text("x,y\n0,0\n1,1\n")

    def test_every_pair_becomes_a_dataset(self, tmp_path):
        from networksynth.configs.gui_config import discover_datasets

        for name in ("gamma", "alpha", "beta"):
            self._pair(tmp_path, name)

        assert discover_datasets(str(tmp_path)) == ["alpha", "beta", "gamma"]

    def test_an_edge_list_without_positions_stops_the_run(self, tmp_path):
        from networksynth.configs.gui_config import discover_datasets

        self._pair(tmp_path, "alpha")
        (tmp_path / "beta_edgelist.csv").write_text("source_index,target_index\n0,1\n")

        with pytest.raises(SpecError, match="has no _positions.csv file beside it"):
            discover_datasets(str(tmp_path))

    def test_a_directory_with_no_pairs_is_rejected(self, tmp_path):
        from networksynth.configs.gui_config import discover_datasets

        (tmp_path / "notes.txt").write_text("nothing to load here")

        with pytest.raises(SpecError, match=r"no \*_edgelist.csv, .* file found"):
            discover_datasets(str(tmp_path))

    def test_the_datasets_reach_the_config(self, tmp_path):
        datasets = tmp_path / "many"
        datasets.mkdir()
        self._pair(datasets, "alpha")
        self._pair(datasets, "beta")

        config = GuiConfig.from_spec(
            _spec(tmp_path, inputs={"datasets_dir": str(datasets)})
        )

        assert [str(d) for d in config.DATASETS] == ["alpha", "beta"]

    def test_each_dataset_resolves_its_own_files(self, tmp_path):
        datasets = tmp_path / "many"
        datasets.mkdir()
        self._pair(datasets, "alpha")
        (datasets / "beta_edgelist.csv").write_text(
            "source_index,target_index\n0,1\n1,2\n"
        )
        (datasets / "beta_positions.csv").write_text("x,y\n0,0\n1,1\n2,2\n")

        config = GuiConfig.from_spec(
            _spec(tmp_path, inputs={"datasets_dir": str(datasets)})
        )
        config.initialize()

        alpha = config.ORIGINAL_NETWORK_FUNC(config.DATASETS[0])
        beta = config.ORIGINAL_NETWORK_FUNC(config.DATASETS[1])

        assert alpha.number_of_nodes() == 2
        assert beta.number_of_nodes() == 3

    def test_a_directory_may_mix_the_three_forms(self, tmp_path):
        import numpy as np
        from scipy.sparse import csr_matrix

        from networksynth.configs.gui_config import discover_datasets
        from networksynth.graphs import read_graph_csv
        from networksynth.graphs.graphml_io import write_graph_graphml

        self._pair(tmp_path, "as_csv")
        np.save(
            tmp_path / "as_npy_adjacency.npy",
            np.array(csr_matrix([[0, 1], [1, 0]]), dtype=object),
        )
        np.save(tmp_path / "as_npy_positions.npy", np.array([[0.0, 0.0], [1.0, 1.0]]))
        write_graph_graphml(
            read_graph_csv(
                str(tmp_path / "as_csv_edgelist.csv"),
                str(tmp_path / "as_csv_positions.csv"),
            ),
            str(tmp_path / "as_graphml_network.graphml"),
        )

        assert discover_datasets(str(tmp_path)) == ["as_csv", "as_graphml", "as_npy"]

    def test_one_prefix_named_twice_stops_the_run(self, tmp_path):
        import numpy as np

        from networksynth.configs.gui_config import discover_datasets

        self._pair(tmp_path, "twice")
        np.save(tmp_path / "twice_adjacency.npy", np.array([[0, 1], [1, 0]]))
        np.save(tmp_path / "twice_positions.npy", np.array([[0.0, 0.0], [1.0, 1.0]]))

        with pytest.raises(SpecError, match="named as two datasets at once"):
            discover_datasets(str(tmp_path))


class TestEntryPointExitCodes:

    def test_bad_spec_exits_2(self, tmp_path):
        assert gui_run.main(["gui_run.py", str(tmp_path / "nope.json")]) == 2

    def test_no_spec_argument_exits_2(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NETWORKSYNTH_RUN_SPEC", raising=False)

        assert gui_run.main(["gui_run.py"]) == 2

    def test_unsupported_mode_exits_2(self, tmp_path):
        assert gui_run.main(["gui_run.py", _spec(tmp_path, mode="not_a_mode")]) == 2

    def test_reads_the_spec_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NETWORKSYNTH_RUN_SPEC", _spec(tmp_path, mode="nope"))

        assert gui_run.main(["gui_run.py"]) == 2

    def test_pipeline_failure_exits_1(self, tmp_path, monkeypatch):
        import networksynth.pipelines.generate as gen

        monkeypatch.setattr(
            gen, "main", lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 1

    def test_cancellation_exits_130(self, tmp_path, monkeypatch):
        import networksynth.pipelines.generate as gen

        monkeypatch.setattr(
            gen, "main", lambda **k: (_ for _ in ()).throw(KeyboardInterrupt)
        )

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 130

    def test_success_exits_0(self, tmp_path, monkeypatch):
        import networksynth.pipelines.generate as gen

        monkeypatch.setattr(gen, "main", lambda **k: None)

        assert gui_run.main(["gui_run.py", _spec(tmp_path)]) == 0


@pytest.mark.integration
@pytest.mark.requires_fixture_data
class TestFullRun:
    def test_spec_in_network_and_manifest_out(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        from networksynth.configs.file_definitions import save_network_csv

        save_network_csv(load_unweighted_test_synth_graph, str(tmp_path / "net.csv"))

        spec = {
            "contract": 2,
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


@pytest.mark.integration
@pytest.mark.requires_fixture_data
class TestDirectoryFullRun:
    def test_one_run_produces_one_output_per_dataset(
        self, tmp_path, load_unweighted_test_synth_graph
    ):
        from networksynth.configs.file_definitions import save_network_csv

        datasets = tmp_path / "inputs"
        datasets.mkdir()
        for name in ("alpha", "beta"):
            save_network_csv(
                load_unweighted_test_synth_graph, str(datasets / f"{name}.csv")
            )

        spec = {
            "contract": 2,
            "mode": "generate",
            "run_name": "unused",
            "output_dir": str(tmp_path / "out"),
            "inputs": {"datasets_dir": str(datasets)},
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
                "SEED": 13,
            },
        }
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        assert gui_run.main(["gui_run.py", str(spec_path)]) == 0

        out = tmp_path / "out"
        roots = [d for d in os.listdir(out) if d.startswith("gui_generate_results_")]
        assert len(roots) == 1, f"a directory is one run, got {roots}"
        root = out / roots[0]

        assert sorted(d for d in os.listdir(root) if (root / d).is_dir()) == [
            "alpha",
            "beta",
        ]
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "ok"
        produced = manifest["outputs"]["edge_lists"]
        assert any(p.startswith("alpha/") for p in produced), produced
        assert any(p.startswith("beta/") for p in produced), produced


@pytest.mark.integration
class TestModeSelection:

    def test_offered_modes_match_dispatchable_modes(self):
        from networksynth.gui.spec_builder import MODES

        assert set(MODES) == set(gui_run._GUI_MODES)

    def test_the_form_asks_for_what_the_spec_requires(self):
        from networksynth.configs.gui_config import MODE_INPUTS
        from networksynth.gui.spec_builder import MODES

        offered = {
            name: tuple(shape.ids for shape in mode.input_shapes)
            for name, mode in MODES.items()
        }
        required = {name: MODE_INPUTS[name] for name in MODES}

        assert offered == required

    @pytest.mark.parametrize("mode", sorted(_MODE_NAMES))
    def test_every_mode_builds_a_valid_spec(self, mode, tmp_path):
        from networksynth.gui import spec_builder

        inputs = _inputs_for(mode, tmp_path)
        values = spec_builder.default_values(mode)
        problems = spec_builder.validate(mode, inputs, str(tmp_path), values)
        assert not problems, problems

        spec = spec_builder.build_spec(mode, inputs, str(tmp_path), values)
        assert spec["mode"] == mode
        assert json.loads(json.dumps(spec)), "spec must survive JSON"

    @pytest.mark.parametrize("mode", sorted(_MODE_NAMES))
    def test_every_mode_rejects_a_missing_input(self, mode, tmp_path):
        from networksynth.gui import spec_builder

        blank = {key: "" for key in spec_builder.default_inputs(mode)}

        problems = spec_builder.validate(
            mode, blank, str(tmp_path), spec_builder.default_values(mode)
        )

        assert problems, f"{mode} accepted an empty input set"

    @pytest.mark.parametrize("mode", sorted(_MODE_NAMES))
    def test_every_mode_spec_loads_back_into_a_config(self, mode, tmp_path):
        from networksynth.gui import spec_builder

        inputs = _inputs_for(mode, tmp_path)
        values = spec_builder.default_values(mode)
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            json.dumps(spec_builder.build_spec(mode, inputs, str(tmp_path), values))
        )

        config = GuiConfig.from_spec(str(spec_path))

        assert config.MODE == mode

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
        from networksynth.gui import spec_builder

        if mode not in _MODE_NAMES:
            pytest.skip(f"{mode} is not currently offered by the GUI")
        required = self._MODE_EXTRAS[mode]

        values = spec_builder.default_values(mode)
        spec = spec_builder.build_spec(
            mode, _inputs_for(mode, tmp_path), str(tmp_path), values
        )
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        config = GuiConfig.from_spec(str(spec_path))

        for name in required:
            assert hasattr(config, name), f"{mode} needs {name}"

    def test_size_params_become_tuples(self, tmp_path):
        from networksynth.gui import spec_builder

        values = spec_builder.default_values("hybrid")
        spec = spec_builder.build_spec(
            "hybrid", _inputs_for("hybrid", tmp_path), str(tmp_path), values
        )
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(spec))

        config = GuiConfig.from_spec(str(spec_path))

        assert isinstance(config.TARGET_SCALE, tuple)
        assert isinstance(config.SYNTHETIC_FRAME_SIZE, tuple)


class TestSpecConfigCrossesProcesses:

    def test_a_spec_built_config_survives_pickling(self, tmp_path):
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert restored.MODE == config.MODE
        assert restored.SEED == config.SEED
        assert restored.BASE_OUTPUT_PATH == config.BASE_OUTPUT_PATH
        assert [str(d) for d in restored.DATASETS] == [str(d) for d in config.DATASETS]

    def test_tuple_params_stay_tuples_after_a_round_trip(self, tmp_path):
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert isinstance(restored.FRAME_SIZE, tuple)
        assert restored.FRAME_SIZE == config.FRAME_SIZE

    def test_the_base_class_still_pickles_by_name(self, tmp_path):
        import pickle

        assert pickle.loads(pickle.dumps(GuiConfig)) is GuiConfig

    def test_the_loader_paths_survive(self, tmp_path):
        import pickle

        config = GuiConfig.from_spec(_spec(tmp_path))

        restored = pickle.loads(pickle.dumps(config))

        assert restored.PATHS["edge_list"] == config.PATHS["edge_list"]
        assert restored.PATHS["positions"] == config.PATHS["positions"]
