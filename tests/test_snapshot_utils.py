import os

import igraph as ig
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _make_synth_graph(n: int = 20, avg_edges: int = 3):
    from graphs.synth_graph import SynthGraph

    positions = np.random.RandomState(42).rand(n, 2) * 100.0
    pairs = [(i, (i + 1) % n) for i in range(n)]
    seen = {(min(u, v), max(u, v)) for u, v in pairs}
    rng = np.random.RandomState(7)
    for _ in range(avg_edges * n // 2):
        u, v = rng.randint(0, n, size=2)
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if u != v and key not in seen:
            seen.add(key)
            pairs.append((int(u), int(v)))
    return SynthGraph(ig.Graph(n=n, edges=pairs), positions)


class TestComputeNetworkMetrics:
    def test_returns_all_keys(self):
        from utils import compute_network_metrics

        graph = _make_synth_graph()
        metrics = compute_network_metrics(graph)
        expected = {
            "node_count",
            "avg_degree",
            "avg_clustering",
            "avg_length",
            "avg_angle",
        }
        assert set(metrics.keys()) == expected

    def test_node_count_matches(self):
        from utils import compute_network_metrics

        graph = _make_synth_graph(n=30)
        metrics = compute_network_metrics(graph)
        assert metrics["node_count"] == graph.number_of_nodes()

    def test_avg_degree_positive(self):
        from utils import compute_network_metrics

        graph = _make_synth_graph()
        metrics = compute_network_metrics(graph)
        assert metrics["avg_degree"] > 0

    def test_values_finite(self):
        from utils import compute_network_metrics

        graph = _make_synth_graph()
        metrics = compute_network_metrics(graph)
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


class TestMetricDistance:
    def test_identical_is_zero(self):
        from utils import metric_distance

        m = {"a": 5.0, "b": 10.0, "c": 3.0}
        assert metric_distance(m, m) == pytest.approx(0.0)

    def test_positive_when_different(self):
        from utils import metric_distance

        ref = {"a": 5.0, "b": 10.0}
        syn = {"a": 6.0, "b": 8.0}
        assert metric_distance(syn, ref) > 0

    def test_handles_zero_ref(self):
        from utils import metric_distance

        ref = {"a": 0.0, "b": 10.0}
        syn = {"a": 2.0, "b": 10.0}
        dist = metric_distance(syn, ref)
        assert dist == pytest.approx(1.0)


def _style(**overrides):
    from configs import RenderStyle

    return RenderStyle(node_size=6.0, line_width=3.0, **overrides)


class TestSaveBfsSnapshot:
    def test_creates_png(self, tmp_path):
        from utils import save_bfs_snapshot

        positions = [(10, 20), (30, 40), (50, 60)]
        edges = {((10, 20), (30, 40)), ((30, 40), (50, 60))}
        frame = ((0, 100), (0, 100))

        save_bfs_snapshot(positions, edges, frame, 0, str(tmp_path), _style(dpi=72))

        expected = os.path.join(str(tmp_path), "snapshot_00000.png")
        assert os.path.isfile(expected), f"PNG not created at {expected}"
        assert os.path.getsize(expected) > 0

    def test_sequential_indices(self, tmp_path):
        from utils import save_bfs_snapshot

        positions = [(5, 5)]
        edges = set()
        frame = ((0, 10), (0, 10))

        for i in range(3):
            save_bfs_snapshot(positions, edges, frame, i, str(tmp_path), _style(dpi=72))

        for i in range(3):
            assert os.path.isfile(os.path.join(str(tmp_path), f"snapshot_{i:05d}.png"))


class TestSaveHybridSnapshot:
    def test_creates_png(self, tmp_path):
        from utils import save_hybrid_snapshot

        positions = [(10, 20), (30, 40)]
        edges = [((10, 20), (30, 40))]
        frame = ((0, 100), (0, 100))

        save_hybrid_snapshot(positions, edges, frame, 0, str(tmp_path), _style(dpi=72))

        expected = os.path.join(str(tmp_path), "snapshot_00000.png")
        assert os.path.isfile(expected)
        assert os.path.getsize(expected) > 0

    def test_empty_graph(self, tmp_path):
        from utils import save_hybrid_snapshot

        save_hybrid_snapshot(
            [], [], ((0, 50), (0, 50)), 0, str(tmp_path), _style(dpi=72)
        )

        expected = os.path.join(str(tmp_path), "snapshot_00000.png")
        assert os.path.isfile(expected)
