# tests/test_pipeline_helpers.py
"""
Unit tests for pure helper functions in the pipeline modules.

Covers:
  - generate.compute_average_error()          → outlier removal & mean
  - generate_select._fmt_metrics()            → metric formatting
  - generate_select._save_metric_report()     → CSV report output
  - hybrid.generate_random_centers()          → Poisson-disk center placement
"""

import csv
import os
import tempfile
from collections import defaultdict
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_average_error  (pipelines/generate.py)
# ---------------------------------------------------------------------------


class TestComputeAverageError:
    def test_empty_list(self):
        from pipelines.generate import compute_average_error

        assert compute_average_error([]) == float("inf")

    def test_all_inf(self):
        from pipelines.generate import compute_average_error

        assert compute_average_error([float("inf"), float("inf")]) == float("inf")

    def test_single_value(self):
        from pipelines.generate import compute_average_error

        assert compute_average_error([0.5]) == pytest.approx(0.5)

    def test_removes_outliers(self):
        from pipelines.generate import compute_average_error

        # 9 values around 1.0 + one massive outlier
        errors = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0, 100.0]
        avg = compute_average_error(errors)
        assert avg < 1.1  # outlier should be removed
        assert avg > 0.9

    def test_mixed_inf_and_valid(self):
        from pipelines.generate import compute_average_error

        errors = [0.1, 0.2, float("inf"), 0.15]
        avg = compute_average_error(errors)
        assert avg != float("inf")
        assert avg > 0


# ---------------------------------------------------------------------------
# _fmt_metrics  (pipelines/generate_select.py)
# ---------------------------------------------------------------------------


class TestFmtMetrics:
    def test_formatting(self):
        from pipelines.generate_select import _fmt_metrics

        m = {
            "node_count": 100,
            "avg_degree": 3.14,
            "avg_clustering": 0.1234,
            "avg_length": 25.67,
            "avg_angle": 120.50,
        }
        s = _fmt_metrics(m)
        assert "nodes=100" in s
        assert "3.14" in s
        assert "0.1234" in s
        assert "25.67" in s
        assert "120.50" in s


# ---------------------------------------------------------------------------
# _save_metric_report  (pipelines/generate_select.py)
# ---------------------------------------------------------------------------


class TestSaveMetricReport:
    def test_creates_csv_with_header_and_rows(self, tmp_path):
        from pipelines.generate_select import _save_metric_report

        ref_metrics = {
            "node_count": 500,
            "avg_degree": 4.0,
            "avg_clustering": 0.3,
            "avg_length": 10.0,
            "avg_angle": 90.0,
        }
        # ranked list: (distance, original_idx, graph_object, metrics_dict, mf_error)
        ranked = [
            (
                0.05,
                0,
                None,
                {
                    "node_count": 480,
                    "avg_degree": 3.8,
                    "avg_clustering": 0.28,
                    "avg_length": 9.5,
                    "avg_angle": 88.0,
                },
                0.12,
            ),
            (
                0.10,
                1,
                None,
                {
                    "node_count": 520,
                    "avg_degree": 4.2,
                    "avg_clustering": 0.32,
                    "avg_length": 10.5,
                    "avg_angle": 92.0,
                },
                0.14,
            ),
        ]

        path = str(tmp_path / "report.csv")
        _save_metric_report(ranked, ref_metrics, path)

        assert os.path.isfile(path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # header + original + 2 ranked = 4 rows
        assert len(rows) == 4
        assert rows[0][0] == "rank"
        assert rows[1][0] == "original"
        assert rows[2][0] == "1"
        assert rows[3][0] == "2"


# ---------------------------------------------------------------------------
# generate_random_centers  (pipelines/hybrid.py)
# ---------------------------------------------------------------------------


class TestGenerateRandomCenters:
    def test_returns_list_of_tuples(self):
        from pipelines.hybrid import generate_random_centers

        centers = generate_random_centers(1000, 1000, min_distance=100, max_centers=10)
        assert isinstance(centers, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in centers)

    def test_respects_max_centers(self):
        from pipelines.hybrid import generate_random_centers

        centers = generate_random_centers(1000, 1000, min_distance=50, max_centers=5)
        assert len(centers) <= 5

    def test_min_distance_constraint(self):
        from pipelines.hybrid import generate_random_centers

        min_dist = 100.0
        centers = generate_random_centers(
            1000, 1000, min_distance=min_dist, max_centers=20
        )
        for i, (x1, y1) in enumerate(centers):
            for j, (x2, y2) in enumerate(centers):
                if i != j:
                    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    assert d >= min_dist - 1e-6

    def test_deterministic_with_seed(self):
        from pipelines.hybrid import generate_random_centers

        c1 = generate_random_centers(500, 500, 50, max_centers=10, rng_seed=42)
        c2 = generate_random_centers(500, 500, 50, max_centers=10, rng_seed=42)
        assert c1 == c2

    def test_zero_max_centers(self):
        from pipelines.hybrid import generate_random_centers

        # max_centers=0 means no limit; should still produce some centers
        centers = generate_random_centers(500, 500, min_distance=50, max_centers=0)
        assert len(centers) > 0

    def test_tight_spacing_limits_count(self):
        from pipelines.hybrid import generate_random_centers

        # Very large min_distance relative to area → few centers
        centers = generate_random_centers(100, 100, min_distance=80, max_centers=0)
        assert len(centers) <= 3  # very hard to fit many


class TestResourceRelease:
    def test_graphnode_reset_reinitializes_spatial_grids(self):
        from graphs._graph_node import GraphNode

        GraphNode.node_grid = defaultdict(set)
        GraphNode.edge_grid = defaultdict(set)
        old_node_grid = GraphNode.node_grid
        old_edge_grid = GraphNode.edge_grid

        GraphNode.reset()

        assert GraphNode.node_grid is not old_node_grid
        assert GraphNode.edge_grid is not old_edge_grid
        assert len(GraphNode.node_grid) == 0
        assert len(GraphNode.edge_grid) == 0
        assert isinstance(GraphNode.node_grid, defaultdict)
        assert isinstance(GraphNode.edge_grid, defaultdict)

    def test_run_hybrid_for_dataset_releases_resources_on_failure(self, monkeypatch):
        from pipelines import hybrid

        class DummyConfig:
            TARGET_SCALE = (1, 1)
            PHASE2_MAX_ROUNDS = 1
            SYNTHETIC_FRAME_SIZE = (1, 1)
            MIN_CENTER_DISTANCE_FACTOR = 1.0
            NUM_CENTERS = 1
            SNAPSHOT_INTERVAL = 0

            @staticmethod
            def get_datasets():
                return [("dummy",)]

        class DummyRunAgent:
            def __init__(self, dataset_id):
                self.dataset_id = dataset_id
                self.attributes = SimpleNamespace(average_degree=2)
                self.mapper = SimpleNamespace(assign_weights=lambda _graph: None)
                self.saver = SimpleNamespace(output_dir=tempfile.gettempdir())

            def prepare_data(self):
                return None

            def get_original_network(self):
                return object()

        class DummyAnalyzer:
            def __init__(self, _graph):
                pass

            def analyze_error_features(self):
                return {}

        reset_calls = []
        close_calls = []
        gc_calls = []

        monkeypatch.setattr(hybrid, "BaseConfig", DummyConfig)
        monkeypatch.setattr(hybrid, "_apply_dataset_factors", lambda _dataset_id: None)
        monkeypatch.setattr(hybrid, "RunAgent", DummyRunAgent)
        monkeypatch.setattr(hybrid, "MultifractalAnalyzer", DummyAnalyzer)
        monkeypatch.setattr(
            hybrid,
            "generate_random_centers",
            lambda *_args, **_kwargs: [(0.0, 0.0)],
        )

        def _fail_phase1(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            hybrid,
            "run_phase1",
            _fail_phase1,
        )
        monkeypatch.setattr(hybrid.GraphNode, "reset", lambda: reset_calls.append(True))
        monkeypatch.setattr(hybrid.plt, "close", lambda arg: close_calls.append(arg))
        monkeypatch.setattr(hybrid.gc, "collect", lambda: gc_calls.append(True) or 0)

        with pytest.raises(RuntimeError, match="boom"):
            hybrid.run_hybrid_for_dataset(("dummy",))

        assert len(reset_calls) == 1
        assert close_calls == ["all"]
        assert len(gc_calls) == 1
