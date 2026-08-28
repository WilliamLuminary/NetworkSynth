import os

import networkit as nk
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _make_graph(n: int = 20, extra_edges: int = 40) -> "SynthGraph":
    from graphs.synth_graph import SynthGraph

    positions = np.random.RandomState(42).rand(n, 2) * 100.0
    g = nk.Graph(n, weighted=False)
    for i in range(n):
        g.addEdge(i, (i + 1) % n)
    rng = np.random.RandomState(7)
    for _ in range(extra_edges):
        u, v = rng.randint(0, n, size=2)
        if u != v and not g.hasEdge(u, v):
            g.addEdge(u, v)
    return SynthGraph(g, positions)


class TestCalculateFrame:
    def test_frame_from_center_position(self):
        from utils import calculate_frame

        frame = calculate_frame(center_position=(50, 50), frame_range=(100, 80))
        assert frame[0] == (0.0, 100.0)
        assert frame[1] == (10.0, 90.0)

    def test_frame_from_graph(self):
        from utils import calculate_frame

        graph = _make_graph(n=10)
        frame = calculate_frame(graph=graph, frame_range=(200, 200))
        positions = graph.positions()
        cx, cy = positions[:, 0].mean(), positions[:, 1].mean()
        assert frame[0][0] == pytest.approx(round(cx - 100, 2))
        assert frame[0][1] == pytest.approx(round(cx + 100, 2))
        assert frame[1][0] == pytest.approx(round(cy - 100, 2))
        assert frame[1][1] == pytest.approx(round(cy + 100, 2))

    def test_no_input_raises(self):
        from utils import calculate_frame

        with pytest.raises(ValueError, match="Either"):
            calculate_frame(frame_range=(100, 100))

    def test_both_inputs_raises(self):
        from utils import calculate_frame

        graph = _make_graph(n=5)
        with pytest.raises(ValueError, match="Only"):
            calculate_frame(graph=graph, center_position=(0, 0), frame_range=(100, 100))


class TestBuildGraph:
    def test_from_edge_list(self):
        from utils import build_graph

        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        graph = build_graph(positions, edges, arg_type="edge_list")
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 4

    def test_from_adjacency_matrix(self):
        from scipy.sparse import csr_matrix

        from utils import build_graph

        n = 5
        positions = np.random.rand(n, 2)
        row = [0, 1, 2, 3, 0]
        col = [1, 2, 3, 4, 4]
        data = [1.0] * 5
        mat = csr_matrix((data, (row, col)), shape=(n, n))
        mat = mat + mat.T
        graph = build_graph(positions, mat, arg_type="adjacency_matrix")
        assert graph.number_of_nodes() >= 2
        assert graph.number_of_edges() >= 1

    def test_unsupported_type_raises(self):
        from utils import build_graph

        with pytest.raises(TypeError, match="Unsupported Type"):
            build_graph(None, arg_type="bad_type")


class TestLargestConnectedComponent:
    def test_connected_graph_unchanged(self):
        from utils import largest_connected_component

        graph = _make_graph(n=10)
        lcc = largest_connected_component(graph)
        assert lcc.number_of_nodes() == graph.number_of_nodes()


class TestRecommendDpi:
    def test_small_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(500) == 150

    def test_medium_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(50_000) == 300

    def test_large_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(500_000) == 600

    def test_very_large_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(5_000_000) == 900

    def test_huge_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(15_000_000) == 1200

    def test_enormous_network(self):
        from utils import recommend_dpi

        assert recommend_dpi(50_000_000) == 1800


class TestFigureToNdarray:
    def test_returns_rgba_array(self):
        from matplotlib.figure import Figure

        from utils import figure_to_ndarray

        fig = Figure(figsize=(2, 2), dpi=50)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        arr = figure_to_ndarray(fig)
        assert arr.ndim == 3
        assert arr.shape[2] == 4
        assert arr.dtype == np.uint8

    def test_swap_channels(self):
        from matplotlib.figure import Figure

        from utils import figure_to_ndarray

        fig = Figure(figsize=(2, 2), dpi=50)
        fig.add_subplot(111)
        normal = figure_to_ndarray(fig, swap_channels=False)
        swapped = figure_to_ndarray(fig, swap_channels=True)
        np.testing.assert_array_equal(normal[..., 0], swapped[..., 2])
        np.testing.assert_array_equal(normal[..., 2], swapped[..., 0])


class TestSaveFigureAsWebp:
    def test_creates_webp_file(self, tmp_path):
        from matplotlib.figure import Figure

        from utils import save_figure_as_webp

        fig = Figure(figsize=(2, 2), dpi=72)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        path = str(tmp_path / "test.webp")
        save_figure_as_webp(fig, path, dpi=72)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_lossy_mode(self, tmp_path):
        from matplotlib.figure import Figure

        from utils import save_figure_as_webp

        fig = Figure(figsize=(2, 2), dpi=72)
        fig.add_subplot(111)
        path = str(tmp_path / "lossy.webp")
        save_figure_as_webp(fig, path, dpi=72, lossless=False)
        assert os.path.isfile(path)


class TestTrimGraph:
    def test_reduces_average_degree(self):
        from utils import trim_graph

        graph = _make_graph(n=50, extra_edges=200)
        original_avg = 2 * graph.number_of_edges() / graph.number_of_nodes()

        target = original_avg * 0.5
        trimmed = trim_graph(graph, target)

        actual_avg = 2 * trimmed.number_of_edges() / trimmed.number_of_nodes()
        assert actual_avg <= 1.1 * target + 0.1

    def test_no_trimming_if_already_below(self):
        from utils import trim_graph

        graph = _make_graph(n=20, extra_edges=5)
        original_edges = graph.number_of_edges()
        current_avg = 2 * original_edges / graph.number_of_nodes()

        trimmed = trim_graph(graph, current_avg * 2)
        assert trimmed.number_of_edges() == original_edges

    def test_empty_graph_handled(self):
        from graphs.synth_graph import SynthGraph
        from utils import trim_graph

        g = nk.Graph(0, weighted=False)
        positions = np.empty((0, 2))
        graph = SynthGraph(g, positions)
        trimmed = trim_graph(graph, 2.0)
        assert trimmed.number_of_nodes() == 0


class TestTimerDecorator:
    def test_timer_returns_result(self):
        from utils import timer

        @timer
        def add(a, b):
            return a + b

        assert add(2, 3) == 5
