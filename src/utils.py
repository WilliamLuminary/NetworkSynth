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


def finalize_plot(fig, show: bool = False) -> np.ndarray:
    """Render a Figure to an ndarray image. Thread-safe (no pyplot globals).

    When *show* is True the rendered image is displayed in a non-blocking
    OpenCV window (requires a GUI environment) instead of calling the
    blocking ``plt.show()``.
    """
    fig.tight_layout(pad=0)
    image = figure_to_ndarray(fig)

    if show:
        try:
            import os

            if os.environ.get("DISPLAY"):
                import cv2

                cv2.imshow("Preview", image[..., :3])
                cv2.waitKey(1)
        except Exception:
            logging.debug("Interactive preview unavailable.")

    # Explicitly close the figure to free memory.
    # Import pyplot only for the close() call; safe because
    # we reference our specific figure, not "current figure".
    from matplotlib import pyplot as _plt

    _plt.close(fig)
    return image


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

    Removes edges in bulk (highest-degree nodes first), then extracts
    the largest connected component.  Repeats if LCC extraction changes
    the node count enough to push the average degree back above target.
    """
    logger = logging.getLogger(__name__)
    target = 1.1 * tar_avg_deg

    round_num = 0
    while True:
        n = graph.number_of_nodes()
        if n == 0:
            return graph
        current_avg = 2 * graph.number_of_edges() / n
        if current_avg <= target:
            break

        target_edges = int(target * n / 2)
        edges_to_remove = graph.number_of_edges() - target_edges

        logger.info(
            f"trim_graph round {round_num}: avg_deg={current_avg:.2f} → "
            f"target≤{target:.2f}, removing ~{edges_to_remove:,} of "
            f"{graph.number_of_edges():,} edges ({n:,} nodes)"
        )

        node_degrees = [(graph.degree(u), u) for u in graph.nodes()]
        node_degrees.sort(reverse=True)

        removed = 0
        for _, u in node_degrees:
            if removed >= edges_to_remove:
                break
            cur_deg = graph.degree(u)
            if cur_deg <= 1:
                continue
            can_remove = min(cur_deg - 1, edges_to_remove - removed)
            nbrs = graph.neighbors(u)[:can_remove]
            for v in nbrs:
                graph.remove_edge(u, v)
                removed += 1

        logger.info(
            f"trim_graph round {round_num}: removed {removed:,} edges, extracting LCC"
        )
        graph = graph.largest_connected_component()
        round_num += 1

    return graph
