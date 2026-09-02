# SPDX-License-Identifier: GPL-3.0-or-later
import csv
import os

import pytest

pytestmark = pytest.mark.unit


class TestComputeAverageError:
    def test_empty_list(self):
        from networksynth.pipelines.generate import compute_average_error

        assert compute_average_error([]) == float("inf")

    def test_all_inf(self):
        from networksynth.pipelines.generate import compute_average_error

        assert compute_average_error([float("inf"), float("inf")]) == float("inf")

    def test_single_value(self):
        from networksynth.pipelines.generate import compute_average_error

        assert compute_average_error([0.5]) == pytest.approx(0.5)

    def test_removes_outliers(self):
        from networksynth.pipelines.generate import compute_average_error

        errors = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99, 1.0, 100.0]
        avg = compute_average_error(errors)
        assert avg < 1.1
        assert avg > 0.9

    def test_mixed_inf_and_valid(self):
        from networksynth.pipelines.generate import compute_average_error

        errors = [0.1, 0.2, float("inf"), 0.15]
        avg = compute_average_error(errors)
        assert avg != float("inf")
        assert avg > 0


class TestFmtMetrics:
    def test_formatting(self):
        from networksynth.pipelines.generate import _fmt_metrics

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


class TestSaveMetricReport:
    def test_creates_csv_with_header_and_rows(self, tmp_path):
        from networksynth.pipelines.generate import _save_metric_report

        ref_metrics = {
            "node_count": 500,
            "avg_degree": 4.0,
            "avg_clustering": 0.3,
            "avg_length": 10.0,
            "avg_angle": 90.0,
        }
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

        assert len(rows) == 4
        assert rows[0][0] == "rank"
        assert rows[1][0] == "original"
        assert rows[2][0] == "1"
        assert rows[3][0] == "2"


class TestGenerateRandomCenters:
    def test_returns_list_of_tuples(self):
        from networksynth.pipelines.hybrid import generate_random_centers

        centers = generate_random_centers(1000, 1000, min_distance=100, max_centers=10)
        assert isinstance(centers, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in centers)

    def test_respects_max_centers(self):
        from networksynth.pipelines.hybrid import generate_random_centers

        centers = generate_random_centers(1000, 1000, min_distance=50, max_centers=5)
        assert len(centers) <= 5

    def test_min_distance_constraint(self):
        from networksynth.pipelines.hybrid import generate_random_centers

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
        from networksynth.pipelines.hybrid import generate_random_centers

        c1 = generate_random_centers(500, 500, 50, max_centers=10, rng_seed=42)
        c2 = generate_random_centers(500, 500, 50, max_centers=10, rng_seed=42)
        assert c1 == c2

    def test_zero_max_centers(self):
        from networksynth.pipelines.hybrid import generate_random_centers

        centers = generate_random_centers(500, 500, min_distance=50, max_centers=0)
        assert len(centers) > 0

    def test_tight_spacing_limits_count(self):
        from networksynth.pipelines.hybrid import generate_random_centers

        # Seeded: unseeded, four corners occasionally fit at min_distance=80
        # (about 1 run in 40), which made this assertion flaky.
        centers = generate_random_centers(
            100, 100, min_distance=80, max_centers=0, rng_seed=0
        )
        assert len(centers) <= 3
