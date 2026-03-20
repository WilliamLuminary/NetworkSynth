# src/utils.py
from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from graphs.synth_graph import SynthGraph


def log_memory(label: str = "") -> None:
    """Log current process RSS memory usage (only when BaseConfig.LOG_MEMORY is True)."""
    from configs import BaseConfig
    from configs.base_config import tagged

    if not BaseConfig.LOG_MEMORY:
        return

    import psutil

    process = psutil.Process()
    rss_gb = process.memory_info().rss / (1024**3)
    logging.getLogger(__name__).info(
        f"{label}: RSS = {rss_gb:.2f} GB",
        extra=tagged("MEMORY"),
    )


def calculate_frame(
    graph: SynthGraph = None,
    center_position: Union[tuple, list, ndarray] = None,
    frame_range: Tuple[int, int] = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if frame_range is None:
        from configs import BaseConfig

        frame_range = BaseConfig.SYNTHETIC_FRAME_SIZE
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
    from graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_sparse_matrix(positions, adjacency_matrix)
    return graph.largest_connected_component()


def _from_graph_node(nodes: set, edges: set) -> SynthGraph:
    from graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_graph_nodes(nodes, edges)
    return graph.largest_connected_component()


def _from_edge_list(positions: np.ndarray, edge_list: np.ndarray) -> SynthGraph:
    from graphs.synth_graph import SynthGraph

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
    (10_000_000, 900),
    (20_000_000, 1200),
]

_DPI_BOOST = 300  # extra DPI headroom for CV2-based rendering


def recommend_dpi(num_nodes: int) -> int:
    """Pick a standard DPI based on network size (matplotlib paths).

    Returns one of 150, 300, 600, 900, 1200, or 1800.
    """
    for threshold, dpi in _DPI_TIERS:
        if num_nodes < threshold:
            return dpi
    return 1800


def recommend_dpi_cv2(num_nodes: int) -> int:
    """Pick DPI for the CV2/OpenCV renderer (+300 over matplotlib tiers).

    Returns one of 450, 600, 900, 1200, 1500, or 2100.
    """
    return recommend_dpi(num_nodes) + _DPI_BOOST


_WEBP_MAX_PX = 16383


def render_network(
    graph: SynthGraph,
    margin_frac: float = 0.02,
    dpi: int = None,
    border: bool = False,
):
    """Render a SynthGraph to a PIL Image using OpenCV.

    Only allocates a fixed-size pixel buffer (H × W × 3 bytes) regardless
    of the number of nodes/edges, avoiding OOM on multi-million-element
    graphs.

    Parameters
    ----------
    graph : SynthGraph
    margin_frac : float
        Fractional margin around the bounding box.
    dpi : int, optional
        Resolution. Auto-selected from node count if omitted.
    border : bool
        If True, draw a thin black rectangle around the bounding box
        (used by the mosaic pipeline).
    """
    import cv2
    from PIL import Image

    if dpi is None:
        dpi = recommend_dpi_cv2(graph.number_of_nodes())

    pos_arr = graph.positions()
    x_min, y_min = pos_arr.min(axis=0)
    x_max, y_max = pos_arr.max(axis=0)
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    x_min -= mx
    x_max += mx
    y_min -= my
    y_max += my

    frame_w = x_max - x_min
    frame_h = y_max - y_min
    # Guard against degenerate bounding boxes (all nodes at same position)
    frame_w = max(frame_w, 1e-12)
    frame_h = max(frame_h, 1e-12)
    aspect = frame_w / frame_h

    img_h = min(int(12 * dpi), _WEBP_MAX_PX)
    img_w = min(int(12 * dpi * aspect), _WEBP_MAX_PX)

    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    sx = (img_w - 1) / frame_w
    sy = (img_h - 1) / frame_h
    px = ((pos_arr[:, 0] - x_min) * sx).astype(np.int32)
    py = ((y_max - pos_arr[:, 1]) * sy).astype(np.int32)

    # Edges (red, batched)
    edge_color = (0, 0, 255)  # BGR
    BATCH = 1_000_000
    edge_list = np.asarray(list(graph.edges()), dtype=np.int64)
    for start in range(0, len(edge_list), BATCH):
        batch = edge_list[start : start + BATCH]
        pts_u = np.column_stack([px[batch[:, 0]], py[batch[:, 0]]])
        pts_v = np.column_stack([px[batch[:, 1]], py[batch[:, 1]]])
        segments = np.stack([pts_u, pts_v], axis=1).astype(np.int32)
        cv2.polylines(canvas, segments, isClosed=False, color=edge_color, thickness=1)
    del edge_list

    # Nodes (blue, direct pixel write)
    node_color_bgr = np.array([255, 0, 0], dtype=np.uint8)
    valid = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    canvas[py[valid], px[valid]] = node_color_bgr

    if border:
        cv2.rectangle(canvas, (0, 0), (img_w - 1, img_h - 1), (0, 0, 0), 2)

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def save_figure_as_webp(fig, filepath: str, *, dpi: int = None, lossless: bool = True):
    """Rasterise a matplotlib Figure and save as WebP via Pillow.

    If the rasterised image exceeds the WebP 16 383-pixel limit in
    either dimension it is downscaled proportionally before encoding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str
    dpi : int, optional
        Override the figure's native DPI for rasterisation.
    lossless : bool
        True for lossless WebP (default), False for lossy (smaller).
    """
    import logging
    from io import BytesIO

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    buf = BytesIO()
    save_kw = {"format": "png", "bbox_inches": "tight"}
    if dpi is not None:
        save_kw["dpi"] = dpi
    fig.savefig(buf, **save_kw)
    buf.seek(0)
    img = Image.open(buf)

    w, h = img.size
    if w > _WEBP_MAX_PX or h > _WEBP_MAX_PX:
        scale = min(_WEBP_MAX_PX / w, _WEBP_MAX_PX / h)
        new_w, new_h = int(w * scale), int(h * scale)
        logging.getLogger(__name__).info(
            f"WebP resize: {w}x{h} -> {new_w}x{new_h} " f"(limit {_WEBP_MAX_PX}px)"
        )
        img = img.resize((new_w, new_h), Image.LANCZOS)

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


def compute_network_metrics(graph: SynthGraph) -> dict:
    """Compute 5 key network metrics for quality comparison.

    Returns a dict with: node_count, avg_degree, avg_clustering,
    avg_length, avg_angle.
    """
    import networkit as nk
    from scipy.spatial.distance import euclidean

    n = graph.number_of_nodes()
    e = graph.number_of_edges()

    node_count = n
    avg_degree = 2.0 * e / n if n > 0 else 0.0

    lcc = nk.centrality.LocalClusteringCoefficient(graph.nk)
    lcc.run()
    avg_clustering = sum(lcc.scores()) / n if n > 0 else 0.0

    positions = graph.positions()
    total_length = 0.0
    for u, v in graph.edges():
        total_length += euclidean(positions[u], positions[v])
    avg_length = total_length / e if e > 0 else 0.0

    all_angle_diffs = []
    for node in graph.nodes():
        nbrs = graph.neighbors(node)
        if len(nbrs) < 2:
            continue
        node_pos = positions[node]
        angles = [
            np.arctan2(
                positions[nbr][1] - node_pos[1],
                positions[nbr][0] - node_pos[0],
            )
            * 180.0
            / np.pi
            for nbr in nbrs
        ]
        sorted_angles = np.sort(angles)
        diffs = np.diff(sorted_angles)
        diffs = np.append(diffs, 360.0 + sorted_angles[0] - sorted_angles[-1])
        all_angle_diffs.extend(diffs)
    avg_angle = float(np.mean(all_angle_diffs)) if all_angle_diffs else 0.0

    return {
        "node_count": node_count,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "avg_length": avg_length,
        "avg_angle": avg_angle,
    }


def metric_distance(metrics: dict, ref_metrics: dict) -> float:
    """Normalised mean relative error across all metrics."""
    total = 0.0
    for key in ref_metrics:
        ref = ref_metrics[key]
        syn = metrics[key]
        total += abs(syn - ref) / abs(ref) if ref != 0 else abs(syn)
    return total / len(ref_metrics)


def save_bfs_snapshot(
    node_positions,
    edges,
    frame,
    index: int,
    output_dir: str,
    *,
    dpi: int = 300,
    node_size: float = 6.0,
    line_width: float = 3.0,
) -> None:
    """Render a BFS snapshot matching the original-network plot style.

    Parameters
    ----------
    node_positions : list of (x, y)
    edges : set of ((x1, y1), (x2, y2))
    frame : ((xmin, xmax), (ymin, ymax))
    index : int
        Snapshot sequence number (used in filename).
    output_dir : str
    dpi : int
    node_size : float
        Marker diameter in points (same unit as ``ax.plot`` markersize).
    line_width : float
        Edge line width in points.
    """
    import os

    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    frame_w = frame[0][1] - frame[0][0]
    frame_h = frame[1][1] - frame[1][0]
    aspect = frame_w / frame_h if frame_h > 0 else 1.0
    fig_h = 10
    fig = Figure(figsize=(fig_h * aspect, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    for edge in edges:
        ax.plot(
            [edge[0][0], edge[1][0]],
            [edge[0][1], edge[1][1]],
            "r-",
            linewidth=line_width,
            zorder=2,
        )

    for pos in node_positions:
        ax.plot(pos[0], pos[1], "bo", markersize=node_size, zorder=3)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])
    ax.add_patch(
        Rectangle(
            (frame[0][0], frame[1][0]),
            frame_w,
            frame_h,
            facecolor="none",
            edgecolor=(0, 0, 0, 0.8),
            linewidth=2,
            zorder=1,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    path = os.path.join(output_dir, f"snapshot_{index:05d}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clear()
    del fig


def save_hybrid_snapshot(
    node_positions,
    edges,
    frame,
    index: int,
    output_dir: str,
    *,
    margin_frac: float = 0.02,
    dpi: int | None = None,
    **_kwargs,
) -> None:
    """Render a snapshot of a large hybrid network and save as PNG.

    Uses the CV2 renderer for consistency with ``render_network`` and
    to avoid OOM on large intermediate snapshots.  Uses fixed *frame*
    limits so all snapshots are aligned for animation.
    """
    import os

    import cv2

    if dpi is None:
        dpi = recommend_dpi_cv2(len(node_positions))

    frame_w = frame[0][1] - frame[0][0]
    frame_h = frame[1][1] - frame[1][0]
    mx = frame_w * margin_frac
    my = frame_h * margin_frac
    aspect = frame_w / frame_h if frame_h > 0 else 1.0

    x_min = frame[0][0] - mx
    x_max = frame[0][1] + mx
    y_min = frame[1][0] - my
    y_max = frame[1][1] + my
    total_w = x_max - x_min
    total_h = y_max - y_min

    img_h = min(int(12 * dpi), _WEBP_MAX_PX)
    img_w = min(int(12 * dpi * aspect), _WEBP_MAX_PX)

    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    sx = (img_w - 1) / total_w
    sy = (img_h - 1) / total_h

    # Draw edges
    if edges:
        edge_arr = np.asarray(
            [((e[0][0], e[0][1]), (e[1][0], e[1][1])) for e in edges],
            dtype=np.float64,
        )
        pts_u = np.column_stack(
            [
                ((edge_arr[:, 0, 0] - x_min) * sx).astype(np.int32),
                ((y_max - edge_arr[:, 0, 1]) * sy).astype(np.int32),
            ]
        )
        pts_v = np.column_stack(
            [
                ((edge_arr[:, 1, 0] - x_min) * sx).astype(np.int32),
                ((y_max - edge_arr[:, 1, 1]) * sy).astype(np.int32),
            ]
        )
        segments = np.stack([pts_u, pts_v], axis=1).astype(np.int32)
        BATCH = 1_000_000
        for start in range(0, len(segments), BATCH):
            cv2.polylines(
                canvas,
                segments[start : start + BATCH],
                isClosed=False,
                color=(0, 0, 255),
                thickness=1,
            )

    # Draw nodes
    if node_positions:
        pos_arr = np.asarray(node_positions, dtype=np.float64)
        px = ((pos_arr[:, 0] - x_min) * sx).astype(np.int32)
        py = ((y_max - pos_arr[:, 1]) * sy).astype(np.int32)
        valid = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
        canvas[py[valid], px[valid]] = np.array([255, 0, 0], dtype=np.uint8)

    path = os.path.join(output_dir, f"snapshot_{index:05d}.png")
    cv2.imwrite(path, canvas)


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

        from graphs.synth_graph import SynthGraph

        graph = SynthGraph(new_nk, graph.positions())
        logger.debug(
            f"trim_graph round {round_num}: rebuilt with "
            f"{graph.number_of_edges():,} edges, extracting LCC"
        )
        graph = graph.largest_connected_component()
        round_num += 1

    return graph
