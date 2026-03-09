# src/utils.py
import functools
import logging
import time
from typing import Tuple, Union

import numpy as np
from numpy import ndarray

from configs import BaseConfig
from graphs.synth_graph import SynthGraph


def calculate_frame(
    graph: SynthGraph = None,
    center_position: Union[tuple, list, ndarray] = None,
    frame_range: Tuple[int, int] = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    frame_range = frame_range or BaseConfig.SYNTHETIC_FRAME_SIZE
    if graph is None:
        if not center_position:
            raise ValueError(
                "Either synthetic_graph or center_position must be provided."
            )
        _center_x, _center_y = center_position
    else:
        if center_position:
            raise ValueError(
                "Only synthetic_graph or center_position must be provided."
            )
        _positions = graph.positions()
        _center_x, _center_y = (
            _positions[:, 0].mean(),
            _positions[:, 1].mean(),
        )

    _half_range = (frame_range[0] / 2, frame_range[1] / 2)
    _width, _height = _half_range

    frame = (
        (
            round(_center_x - _width, 2),
            round(_center_x + _width, 2),
        ),
        (
            round(_center_y - _height, 2),
            round(_center_y + _height, 2),
        ),
    )
    return frame


def build_graph(*args, arg_type: str = "adjacency_matrix") -> SynthGraph:
    if arg_type == "adjacency_matrix":
        return _from_adjacency_matrix(*args)
    elif arg_type == "graph_node":
        return _from_graph_node(*args)
    elif arg_type == "edge_list":
        return _from_edge_list(*args)
    else:
        raise TypeError(
            f"Unsupported Type: {arg_type} " f"from function {build_graph.__name__}."
        )


def _from_adjacency_matrix(positions: np.ndarray, adjacency_matrix) -> SynthGraph:
    graph = SynthGraph.from_sparse_matrix(positions, adjacency_matrix)
    return graph.largest_connected_component()


def _from_graph_node(nodes: set, edges: set) -> SynthGraph:
    graph = SynthGraph.from_graph_nodes(nodes, edges)
    return graph.largest_connected_component()


def _from_edge_list(positions: np.ndarray, edge_list: np.ndarray) -> SynthGraph:
    graph = SynthGraph.from_edge_list(positions, edge_list)
    return graph.largest_connected_component()


def largest_connected_component(graph: SynthGraph) -> SynthGraph:
    """
    Keep only the largest connected component of the graph.
    """
    return graph.largest_connected_component()


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(
            "Time taken by func %r: %.2f seconds",
            func.__name__,
            end - start,
        )
        return result

    return wrapper


def finalize_plot(fig, show: bool = False):
    """Finalise a matplotlib Figure for saving.

    Returns the *Figure* itself so callers can persist it as SVG (or any
    other vector format) without rasterisation.  When *show* is True the
    figure is temporarily rasterised for an interactive OpenCV preview.
    """
    fig.tight_layout(pad=0)

    if show:
        try:
            import os

            if os.environ.get("DISPLAY"):
                import cv2

                preview = figure_to_ndarray(fig)
                cv2.imshow("Preview", preview[..., :3])
                cv2.waitKey(1)
        except Exception:
            logging.debug("Interactive preview unavailable.")

    return fig


_DPI_TIERS = [
    (10_000, 150),
    (100_000, 300),
    (1_000_000, 600),
    (5_000_000, 900),
    (20_000_000, 1200),
]


def recommend_dpi(num_nodes: int) -> int:
    """Pick a standard DPI based on network size.

    Returns one of 150, 300, 600, 900, 1200, or 1800.
    """
    for threshold, dpi in _DPI_TIERS:
        if num_nodes < threshold:
            return dpi
    return 1800


def save_figure_as_webp(fig, filepath: str, *, dpi: int = None, lossless: bool = True):
    """Rasterise a matplotlib Figure and save as WebP via Pillow.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str
    dpi : int, optional
        Override the figure's native DPI for rasterisation.
    lossless : bool
        True for lossless WebP (default), False for lossy (smaller).
    """
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    save_kw = {"format": "png", "bbox_inches": "tight"}
    if dpi is not None:
        save_kw["dpi"] = dpi
    fig.savefig(buf, **save_kw)
    buf.seek(0)
    img = Image.open(buf)
    img.save(filepath, "webp", lossless=lossless)


def figure_to_ndarray(fig, swap_channels: bool = False) -> ndarray:
    """Render a matplotlib Figure to an RGBA ndarray via the Agg canvas.

    This is thread-safe: it attaches a fresh FigureCanvasAgg to the
    figure and never touches pyplot global state.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image_array = np.asarray(buf)
    if swap_channels:
        image_array = image_array[..., [2, 1, 0, 3]]
    return image_array


def trim_graph(graph: SynthGraph, tar_avg_deg: float) -> SynthGraph:
    """Trim edges from high-degree nodes until average degree <= 1.1 * target.

    Uses vectorised NumPy to score all edges and ``argpartition`` to
    select which to keep in O(E), then rebuilds the graph and extracts
    the LCC.  Repeats if LCC shifts the average degree above target.
    """
    import networkit as nk

    logger = logging.getLogger(__name__)
    target = 1.1 * tar_avg_deg

    round_num = 0
    while True:
        n = graph.number_of_nodes()
        if n == 0:
            return graph
        m = graph.number_of_edges()
        current_avg = 2 * m / n
        if current_avg <= target:
            break

        target_edges = int(target * n / 2)
        logger.debug(
            f"trim_graph round {round_num}: avg_deg={current_avg:.2f} → "
            f"target≤{target:.2f}, keeping {target_edges:,} of "
            f"{m:,} edges ({n:,} nodes)"
        )

        src = np.empty(m, dtype=np.int64)
        dst = np.empty(m, dtype=np.int64)
        for i, (u, v) in enumerate(graph.edges()):
            src[i] = u
            dst[i] = v

        degrees = np.array([graph.degree(u) for u in range(n)], dtype=np.int32)
        edge_scores = np.maximum(degrees[src], degrees[dst])
        keep_idx = np.argpartition(edge_scores, target_edges)[:target_edges]

        kept_src = src[keep_idx]
        kept_dst = dst[keep_idx]
        weighted = graph.is_weighted()
        new_nk = nk.Graph(n, weighted=weighted)
        if weighted:
            for i in range(len(kept_src)):
                new_nk.addEdge(
                    int(kept_src[i]),
                    int(kept_dst[i]),
                    graph.weight(int(kept_src[i]), int(kept_dst[i])),
                )
        else:
            for i in range(len(kept_src)):
                new_nk.addEdge(int(kept_src[i]), int(kept_dst[i]))

        graph = SynthGraph(new_nk, graph.positions())
        logger.debug(
            f"trim_graph round {round_num}: rebuilt with "
            f"{graph.number_of_edges():,} edges, extracting LCC"
        )
        graph = graph.largest_connected_component()
        round_num += 1

    return graph
