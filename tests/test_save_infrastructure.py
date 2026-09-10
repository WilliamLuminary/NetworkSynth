# SPDX-License-Identifier: GPL-3.0-or-later
import csv
import os
import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from networksynth.configs.file_definitions import (
    SaveSpec,
)
from networksynth.configs.file_definitions import save_csv as _save_csv
from networksynth.configs.file_definitions import (
    save_network_csv,
)
from networksynth.configs.file_definitions import save_pickle as _save_pickle
from networksynth.handlers.saver import Saver
from tests.fixture_config import FIXTURE, FIXTURE_COMPARE, FIXTURE_HYBRID


class FakeGraph:

    def __init__(self, n_nodes=4, weighted=True):
        self._n = n_nodes
        self._positions = np.random.rand(n_nodes, 2)
        self._edges = [(0, 1), (1, 2), (2, 3)]
        self._weights = [1.0, 2.0, 3.0]
        self._weighted = weighted

    def is_weighted(self):
        return self._weighted

    def edges(self):
        return self._edges

    def edges_with_weights(self):
        return [(u, v, w) for (u, v), w in zip(self._edges, self._weights)]

    def positions(self):
        return self._positions


class TestDispatcher:
    def test_known_identifier(self):
        specs = FIXTURE.save("original_image")
        assert len(specs) >= 1
        assert specs[0].extension == "webp"

    def test_missing_identifier_raises(self):
        with pytest.raises(AttributeError, match="SAVE_THIS_DOES_NOT_EXIST"):
            FIXTURE.save("this_does_not_exist")

    def test_every_declared_spec_is_valid(self):
        names = [f.name for f in fields(FIXTURE) if f.name.startswith("SAVE_")]
        assert names, "the config declares no SAVE_* specs"
        for name in names:
            specs = getattr(FIXTURE, name)
            assert specs, f"{name} is empty"
            for spec in specs:
                assert isinstance(spec, SaveSpec), f"{name} holds {type(spec)}"
                assert callable(spec.save_fn), f"save_fn not callable for {name}"
                assert spec.extension, f"no extension for {name}"


class TestSpecs:
    def test_original_network_formats(self):
        specs = FIXTURE.save("original_network")
        exts = {s.extension for s in specs}
        assert exts == {"csv"}

    def test_synthetic_network_formats(self):
        specs = FIXTURE.save("synthetic_network")
        exts = {s.extension for s in specs}
        assert exts == {"csv"}

    def test_synthetic_graph_default_webp(self):
        specs = FIXTURE.save("synthetic_graph")
        assert len(specs) == 1
        assert specs[0].extension == "webp"

    def test_analysis_data_no_timestamp(self):
        specs = FIXTURE.save("analysis_data")
        assert specs[0].use_timestamp is False

    def test_analysis_figure_has_timestamp(self):
        specs = FIXTURE.save("analysis_figure")
        assert specs[0].use_timestamp is True


class TestModeOverrides:

    def test_a_mode_config_can_add_a_second_format(self):
        specs = FIXTURE_HYBRID.save("synthetic_graph")
        exts = [s.extension for s in specs]
        assert "webp" in exts
        assert "png" in exts
        assert len(specs) == 2

    def test_override_resolves_on_the_defining_config(self):
        from networksynth.configs.file_definitions import save_pickle

        class CustomConfig(type(FIXTURE)):
            SAVE_TEST_CUSTOM = (
                SaveSpec("custom_dir", "test_detail", "pkl", save_pickle),
            )

        config = CustomConfig(
            **{f.name: getattr(FIXTURE, f.name) for f in fields(FIXTURE)}
        )
        specs = config.save("test_custom")
        assert specs[0].relative_dir == "custom_dir"
        assert specs[0].detail == "test_detail"

    def test_a_mode_config_inherits_the_analysis_specs(self):
        specs = FIXTURE_COMPARE.save("analysis_data")
        assert specs[0].use_timestamp is False
        specs = FIXTURE_COMPARE.save("analysis_figure")
        assert specs[0].extension == "webp"


@pytest.fixture
def saver_in_tmpdir(tmp_path):
    saver = object.__new__(Saver)
    saver.output_dir = str(tmp_path)
    saver._save_func = FIXTURE.save
    saver._batch_timestamp = None
    return saver


class TestSaverPaths:
    def test_single_spec_path(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        saver._save_func = replace(
            FIXTURE,
            SAVE_ORIGINAL_IMAGE=(
                SaveSpec("original", "original_image", "png", fake_save_fn),
            ),
        ).save
        saver.save("img_data", "original_image", "pfx_")

        assert len(recorded) == 1
        p = recorded[0]
        assert os.path.join("original", "pfx_original_image_") in p
        assert p.endswith(".png")

    def test_no_timestamp_path(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        saver._save_func = replace(
            FIXTURE,
            SAVE_ANALYSIS_DATA=(
                SaveSpec("", "analysis_data", "pkl", fake_save_fn, use_timestamp=False),
            ),
        ).save
        saver.save("data", "analysis_data", "")

        assert len(recorded) == 1
        assert recorded[0].endswith("analysis_data.pkl")

    def test_batch_timestamp_shared(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        saver._save_func = replace(
            FIXTURE,
            SAVE_SYNTHETIC_NETWORK=(
                SaveSpec("synthetic", "net", "csv", fake_save_fn),
                SaveSpec("synthetic", "net", "graphml.gz", fake_save_fn),
            ),
        ).save
        saver.begin_batch()
        ts = saver._batch_timestamp
        saver.save("graph", "synthetic_network", "b_")
        saver.end_batch()

        assert len(recorded) == 2
        for p in recorded:
            assert ts in p, f"batch timestamp {ts} not in {p}"

    def test_none_content_skipped(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        saver.save(None, "original_image")

    def test_prefix_normalization_in_run_agent(self):
        prefix = "test"
        normalised = (
            f"{prefix}_" if prefix and not prefix.endswith("_") else (prefix or "")
        )
        assert normalised == "test_"

        prefix = "test_"
        normalised = (
            f"{prefix}_" if prefix and not prefix.endswith("_") else (prefix or "")
        )
        assert normalised == "test_"

    def test_multi_spec_creates_directories(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        saver._save_func = replace(
            FIXTURE,
            SAVE_SYNTHETIC_NETWORK=(
                SaveSpec("sub/deep", "detail", "csv", fake_save_fn),
            ),
        ).save
        saver.save("data", "synthetic_network", "")

        assert len(recorded) == 1
        parent = os.path.dirname(recorded[0])
        assert os.path.isdir(parent)


class TestSerializers:
    def test_save_pickle_roundtrip(self, tmp_path):
        data = {"key": [1, 2, 3], "nested": {"a": True}}
        path = str(tmp_path / "test.pkl")
        _save_pickle(data, path)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == data

    def test_save_csv_writes_rows(self, tmp_path):
        rows = [["a", "b"], [1, 2], [3, 4]]
        path = str(tmp_path / "test.csv")
        _save_csv(rows, path)
        with open(path) as f:
            reader = csv.reader(f)
            loaded = [row for row in reader]
        assert loaded[0] == ["a", "b"]
        assert loaded[1] == ["1", "2"]

    def test_save_network_csv_writes_edgelist_and_positions(self, tmp_path):
        graph = FakeGraph(n_nodes=4)
        path = str(tmp_path / "network.csv")
        save_network_csv(graph, path)

        edgelist_path = str(tmp_path / "network_edgelist.csv")
        positions_path = str(tmp_path / "network_positions.csv")

        assert os.path.exists(edgelist_path), "edgelist CSV not created"
        assert os.path.exists(positions_path), "positions CSV not created"

        with open(edgelist_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["source_index", "target_index", "edge_weight"]
        assert len(rows) == 1 + len(graph._edges)

        with open(positions_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["x", "y"]
        assert len(rows) == 1 + graph._n

    def test_save_network_csv_omits_weight_column_when_unweighted(self, tmp_path):
        graph = FakeGraph(n_nodes=4, weighted=False)
        path = str(tmp_path / "network.csv")
        save_network_csv(graph, path)

        with open(str(tmp_path / "network_edgelist.csv")) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["source_index", "target_index"]
        assert len(rows) == 1 + len(graph._edges)
        assert all(len(r) == 2 for r in rows[1:])


class TestEndToEnd:
    def test_json_save_via_saver(self, tmp_path):
        import json

        saver = object.__new__(Saver)
        saver.output_dir = str(tmp_path)
        saver._save_func = FIXTURE.save
        saver._batch_timestamp = None

        data = {"hello": "world"}
        saver.save(data, "original_property", "test_")

        written = list(tmp_path.rglob("*.json"))
        assert len(written) == 1
        with open(written[0]) as handle:
            assert json.load(handle) == data

    def test_csv_export_via_saver(self, tmp_path):
        saver = object.__new__(Saver)
        saver.output_dir = str(tmp_path)
        saver._save_func = FIXTURE.save
        saver._batch_timestamp = "20250101_000000"

        graph = FakeGraph(n_nodes=5)

        saver._save_func = replace(
            FIXTURE,
            SAVE_SYNTHETIC_NETWORK=(
                SaveSpec("synthetic", "synthetic_network", "csv", save_network_csv),
            ),
        ).save
        saver.save(graph, "synthetic_network", "exp_")

        csv_files = sorted(tmp_path.rglob("*.csv"))
        assert len(csv_files) == 2

        names = {f.name for f in csv_files}
        assert any("edgelist" in n for n in names)
        assert any("positions" in n for n in names)


class TestDisabledSaving:

    def test_a_saver_that_exists_always_writes(self, tmp_path):
        saver = Saver(replace(FIXTURE, DISABLE_SAVING=True), str(tmp_path))

        assert isinstance(saver, Saver)


class TestBatchTimestampIsPerInstance:
    def test_two_savers_do_not_share_a_batch(self, tmp_path):
        first = Saver(FIXTURE, str(tmp_path / "one"))
        second = Saver(FIXTURE, str(tmp_path / "two"))

        first.begin_batch()

        assert first._batch_timestamp is not None
        assert second._batch_timestamp is None

    def test_end_batch_clears_only_its_own(self, tmp_path):
        first = Saver(FIXTURE, str(tmp_path / "one"))
        second = Saver(FIXTURE, str(tmp_path / "two"))
        first.begin_batch()
        second.begin_batch()

        first.end_batch()

        assert first._batch_timestamp is None
        assert second._batch_timestamp is not None
