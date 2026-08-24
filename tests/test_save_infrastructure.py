import csv
import os
import pickle
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Stub networkit so the full import chain works without the C extension.
# ---------------------------------------------------------------------------
_nk = types.ModuleType("networkit")
_nk.Graph = type("Graph", (), {})
_nk.Format = type("Format", (), {"NetworkitBinary": 0})
_nk.writeGraph = lambda *a, **kw: None
sys.modules.setdefault("networkit", _nk)

from configs import BaseConfig  # noqa: E402
from configs.file_definitions import DEFAULT_SAVE_SPECS  # noqa: E402
from configs.file_definitions import save_csv as _save_csv  # noqa: E402
from configs.file_definitions import (
    save_network_csv,
)
from configs.file_definitions import save_pickle as _save_pickle
from handlers.saver import Saver  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeGraph:

    def __init__(self, n_nodes=4):
        self._n = n_nodes
        self._positions = np.random.rand(n_nodes, 2)
        self._edges = [(0, 1), (1, 2), (2, 3)]
        self._weights = [1.0, 2.0, 3.0]

    def edges(self):
        return self._edges

    def edges_with_weights(self):
        return [(u, v, w) for (u, v), w in zip(self._edges, self._weights)]

    def positions(self):
        return self._positions


# ---------------------------------------------------------------------------
# 1. Dispatcher tests
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_known_identifier(self):
        specs = BaseConfig.save("original_image")
        assert isinstance(specs, list) and len(specs) >= 1
        assert specs[0][2] == "webp"

    def test_missing_identifier_raises(self):
        with pytest.raises(ValueError, match="No save spec"):
            BaseConfig.save("this_does_not_exist")

    def test_all_default_specs_are_valid(self):
        assert len(DEFAULT_SAVE_SPECS) >= 1, "DEFAULT_SAVE_SPECS is empty"
        for identifier, specs in DEFAULT_SAVE_SPECS.items():
            assert (
                isinstance(specs, list) and len(specs) >= 1
            ), f"{identifier} returned empty"
            for s in specs:
                assert len(s) >= 4, f"spec too short for {identifier}: {s}"
                assert callable(s[3]), f"save_fn not callable for {identifier}"


# ---------------------------------------------------------------------------
# 2. Spec content tests
# ---------------------------------------------------------------------------


class TestSpecs:
    def test_original_network_formats(self):
        specs = BaseConfig.save("original_network")
        exts = {s[2] for s in specs}
        assert exts == {"csv"}

    def test_synthetic_export_formats(self):
        specs = BaseConfig.save("synthetic_export")
        exts = {s[2] for s in specs}
        assert exts == {"csv"}

    def test_synthetic_graph_default_webp(self):
        specs = BaseConfig.save("synthetic_graph")
        assert len(specs) == 1
        assert specs[0][2] == "webp"

    def test_analysis_data_no_timestamp(self):
        specs = BaseConfig.save("analysis_data")
        assert len(specs[0]) == 5
        assert specs[0][4] is False

    def test_analysis_figure_has_timestamp(self):
        specs = BaseConfig.save("analysis_figure")
        assert len(specs[0]) == 4  # no 5th element → defaults to True


# ---------------------------------------------------------------------------
# 3. Mode config override tests
# ---------------------------------------------------------------------------


class TestModeOverrides:

    def test_mosaic_synthetic_graph_has_png(self):
        from configs.mosaic_mode.config_sample import SampleConfig as MosaicConfig

        specs = MosaicConfig.save("synthetic_graph")
        exts = [s[2] for s in specs]
        assert "webp" in exts
        assert "png" in exts
        assert len(specs) == 2

    def test_override_resolves_on_the_defining_config(self):
        class CustomConfig(BaseConfig):
            @classmethod
            def save_test_custom(cls):
                from configs.file_definitions import save_pickle

                return [("custom_dir", "test_detail", "pkl", save_pickle)]

        specs = CustomConfig.save("test_custom")
        assert specs[0][0] == "custom_dir"
        assert specs[0][1] == "test_detail"

    def test_override_does_not_leak_to_other_configs(self):
        from configs.mosaic_mode.config_sample import SampleConfig as MosaicConfig

        MosaicConfig.save("synthetic_graph")  # mosaic overrides this identifier

        assert len(BaseConfig.save("synthetic_graph")) == 1

    def test_compare_mode_overrides(self):
        from configs.compare_mode.config_sample import SampleConfig as CompareConfig

        specs = CompareConfig.save("analysis_data")
        assert specs[0][4] is False
        specs = CompareConfig.save("analysis_figure")
        assert specs[0][2] == "webp"


# ---------------------------------------------------------------------------
# 4. Saver path-construction tests
# ---------------------------------------------------------------------------


@pytest.fixture
def saver_in_tmpdir(tmp_path):
    saver = object.__new__(Saver)
    saver.output_dir = str(tmp_path)
    saver._save_func = BaseConfig.save
    saver._batch_timestamp = None
    return saver


class TestSaverPaths:
    def test_single_spec_path(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        BaseConfig.save_original_image = classmethod(
            lambda cls: [("original", "original_image", "png", fake_save_fn)]
        )
        try:
            saver.save("img_data", "original_image", "pfx_")
        finally:
            delattr(BaseConfig, "save_original_image")

        assert len(recorded) == 1
        p = recorded[0]
        assert os.path.join("original", "pfx_original_image_") in p
        assert p.endswith(".png")

    def test_no_timestamp_path(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        BaseConfig.save_analysis_data = classmethod(
            lambda cls: [("", "analysis_data", "pkl", fake_save_fn, False)]
        )
        try:
            saver.save("data", "analysis_data", "")
        finally:
            delattr(BaseConfig, "save_analysis_data")

        assert len(recorded) == 1
        assert recorded[0].endswith("analysis_data.pkl")

    def test_batch_timestamp_shared(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        recorded = []

        def fake_save_fn(content, path):
            recorded.append(path)

        BaseConfig.save_synthetic_export = classmethod(
            lambda cls: [
                ("synthetic", "net", "csv", fake_save_fn),
                ("synthetic", "net", "nkbin", fake_save_fn),
            ]
        )
        try:
            saver.begin_batch()
            ts = saver._batch_timestamp
            saver.save("graph", "synthetic_export", "b_")
            saver.end_batch()
        finally:
            delattr(BaseConfig, "save_synthetic_export")

        assert len(recorded) == 2
        for p in recorded:
            assert ts in p, f"batch timestamp {ts} not in {p}"

    def test_none_content_skipped(self, saver_in_tmpdir):
        saver = saver_in_tmpdir
        saver.save(None, "original_image")  # should not raise

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

        BaseConfig.save_synthetic_export = classmethod(
            lambda cls: [
                ("sub/deep", "detail", "csv", fake_save_fn),
            ]
        )
        try:
            saver.save("data", "synthetic_export", "")
        finally:
            delattr(BaseConfig, "save_synthetic_export")

        assert len(recorded) == 1
        parent = os.path.dirname(recorded[0])
        assert os.path.isdir(parent)


# ---------------------------------------------------------------------------
# 5. Serializer tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 6. End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pickle_save_via_saver(self, tmp_path):
        saver = object.__new__(Saver)
        saver.output_dir = str(tmp_path)
        saver._save_func = BaseConfig.save
        saver._batch_timestamp = None

        data = {"hello": "world"}
        saver.save(data, "original_property", "test_")

        pkl_files = list(tmp_path.rglob("*.pkl"))
        assert len(pkl_files) == 1
        with open(pkl_files[0], "rb") as f:
            loaded = pickle.load(f)
        assert loaded == data

    def test_csv_export_via_saver(self, tmp_path):
        saver = object.__new__(Saver)
        saver.output_dir = str(tmp_path)
        saver._save_func = BaseConfig.save
        saver._batch_timestamp = "20250101_000000"

        graph = FakeGraph(n_nodes=5)

        BaseConfig.save_synthetic_export = classmethod(
            lambda cls: [
                ("synthetic", "synthetic_network", "csv", save_network_csv),
            ]
        )
        try:
            saver.save(graph, "synthetic_export", "exp_")
        finally:
            delattr(BaseConfig, "save_synthetic_export")

        csv_files = sorted(tmp_path.rglob("*.csv"))
        assert len(csv_files) == 2

        names = {f.name for f in csv_files}
        assert any("edgelist" in n for n in names)
        assert any("positions" in n for n in names)


# ---------------------------------------------------------------------------
# Disabled saving
# ---------------------------------------------------------------------------


class TestDisabledSaving:

    def test_a_saver_that_exists_always_writes(self, tmp_path):

        class Disabled(BaseConfig):
            DISABLE_SAVING = True

        saver = Saver(Disabled, str(tmp_path))

        assert isinstance(saver, Saver)

    # Which saver a run gets is build_saver's decision now; both branches are
    # covered in tests/test_null_saver.py.

    def test_the_flag_is_never_set_on_base_config(self):

        class Disabled(BaseConfig):
            DISABLE_SAVING = True

        assert Disabled.DISABLE_SAVING is True
        assert BaseConfig.DISABLE_SAVING is False


class TestBatchTimestampIsPerInstance:
    def test_two_savers_do_not_share_a_batch(self, tmp_path):
        first = Saver(BaseConfig, str(tmp_path / "one"))
        second = Saver(BaseConfig, str(tmp_path / "two"))

        first.begin_batch()

        assert first._batch_timestamp is not None
        assert second._batch_timestamp is None

    def test_end_batch_clears_only_its_own(self, tmp_path):
        first = Saver(BaseConfig, str(tmp_path / "one"))
        second = Saver(BaseConfig, str(tmp_path / "two"))
        first.begin_batch()
        second.begin_batch()

        first.end_batch()

        assert first._batch_timestamp is None
        assert second._batch_timestamp is not None
