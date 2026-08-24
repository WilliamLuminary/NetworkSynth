from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from .graph_ops import calculate_frame

if TYPE_CHECKING:
    from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


def finalize_plot(fig, show: bool = False):
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
    for threshold, dpi in _DPI_TIERS:
        if num_nodes < threshold:
            return dpi
    return 1800


def _points_to_px(points: float | None, dpi: int, *, minimum: int) -> int:
    """Points to pixels at *dpi*, floored at *minimum*.

    A point is 1/72 inch, and these canvases are 12 inches tall, matching the
    matplotlib renderers so one number means one thing across all of them.
    """
    if points is None:
        return minimum
    return max(minimum, round(points * dpi / 72))


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
    max_px: int | None = None,
):
    """Render a SynthGraph to a BGR ndarray using OpenCV.

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

    # Max pixel size of the render, always capped at the WebP 16383px hard
    # limit. A large network at that limit produces a ~190MP image most viewers
    # cannot open, so callers can pass a lower max_px.
    max_px = min(max_px or _WEBP_MAX_PX, _WEBP_MAX_PX)
    img_h = int(12 * dpi)
    img_w = int(12 * dpi * aspect)
    # Clamp to max_px while preserving aspect: when either axis exceeds the
    # cap, scale both by the same factor (clamping each axis independently
    # would square a rectangular network).
    clamp = min(1.0, max_px / max(img_h, img_w))
    img_h = max(1, int(img_h * clamp))
    img_w = max(1, int(img_w * clamp))

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

    return canvas


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
    max_px: int | None = None,
    node_size: float | None = None,
    line_width: float | None = None,
) -> None:
    """Render one hybrid Phase 2 snapshot with OpenCV.

    ``node_size`` (marker diameter) and ``line_width`` are in points, the same
    unit as :func:`save_bfs_snapshot` and matplotlib, and are converted to
    pixels at *dpi*.  Unset means the thinnest the renderer can draw: a 1px line
    and a single-pixel node.

    No ``**kwargs``: an unsupported key is a config asking for something this
    renderer cannot do, and it used to be swallowed silently.
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

    # Max pixel size of the render, always capped at the WebP 16383px hard
    # limit. A large network at that limit produces a ~190MP image most viewers
    # cannot open, so callers can pass a lower max_px.
    max_px = min(max_px or _WEBP_MAX_PX, _WEBP_MAX_PX)
    img_h = int(12 * dpi)
    img_w = int(12 * dpi * aspect)
    # Clamp to max_px while preserving aspect: when either axis exceeds the
    # cap, scale both by the same factor (clamping each axis independently
    # would square a rectangular network).
    clamp = min(1.0, max_px / max(img_h, img_w))
    img_h = max(1, int(img_h * clamp))
    img_w = max(1, int(img_w * clamp))

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
        thickness = _points_to_px(line_width, dpi, minimum=1)
        for start in range(0, len(segments), BATCH):
            cv2.polylines(
                canvas,
                segments[start : start + BATCH],
                isClosed=False,
                color=(0, 0, 255),
                thickness=thickness,
            )

    # Draw nodes
    if node_positions:
        pos_arr = np.asarray(node_positions, dtype=np.float64)
        px = ((pos_arr[:, 0] - x_min) * sx).astype(np.int32)
        py = ((y_max - pos_arr[:, 1]) * sy).astype(np.int32)
        valid = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
        node_color = np.array([255, 0, 0], dtype=np.uint8)
        radius = _points_to_px(node_size, dpi, minimum=0) // 2
        if radius < 1:
            # A single pixel per node: cheaper than a circle of radius 0, and
            # the same result.
            canvas[py[valid], px[valid]] = node_color
        else:
            for x, y in zip(px[valid], py[valid]):
                cv2.circle(canvas, (int(x), int(y)), radius, (255, 0, 0), thickness=-1)

    path = os.path.join(output_dir, f"snapshot_{index:05d}.png")
    cv2.imwrite(path, canvas)


def plot_network(
    data_type: str,
    graph: SynthGraph,
    node_scale: float = 1.0,
    frame_size=None,
    synthetic_frame_size=None,
    node_size: float | None = None,
    line_width: float | None = None,
    **kwargs,
):
    """Plot a network with matplotlib.

    ``node_size`` and ``line_width`` (points) override the identifier's
    :class:`PlotConfig` for this call.  Passed per call rather than written into
    the shared PlotConfig, which would restyle every plot in the process.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    from configs import FILE_CONFIGURATIONS, PlotConfig

    file_config = FILE_CONFIGURATIONS.get(data_type)
    assert isinstance(
        file_config, PlotConfig
    ), f"{data_type} shouldn't call plot_network."

    if data_type == "original_graph":
        assert frame_size is not None, "original_graph requires frame_size"
        frame = (
            (0, frame_size[0]),
            (0, frame_size[1]),
        )
    else:
        assert (
            synthetic_frame_size is not None
        ), f"{data_type} requires synthetic_frame_size"
        frame = calculate_frame(graph, frame_range=synthetic_frame_size)

    frame_width = frame[0][1] - frame[0][0]
    frame_height = frame[1][1] - frame[1][0]
    aspect_ratio = frame_width / frame_height
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig = Figure(figsize=(fig_width, fig_height), dpi=300)
    ax = fig.add_subplot(111)

    positions = graph.positions()
    edge_width = file_config.line_width if line_width is None else line_width
    for u, v in graph.edges():
        pos_u = positions[u]
        pos_v = positions[v]
        ax.plot(
            [pos_u[0], pos_v[0]],
            [pos_u[1], pos_v[1]],
            "r-",
            linewidth=edge_width,
            zorder=2,
        )

    node_scale = node_scale if data_type == "original_graph" else 1.0
    base_node_size = file_config.node_size if node_size is None else node_size
    marker_size = base_node_size * node_scale
    for node in graph.nodes():
        pos = positions[node]
        ax.plot(pos[0], pos[1], "bo", markersize=marker_size, zorder=2)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type == "original_graph":
        image = kwargs.get("background", None)
        if image is not None:
            alpha = file_config.alpha
            img_height, img_width = image.shape[:2]
            ax.imshow(
                image,
                cmap="gray",
                alpha=alpha,
                extent=(0, img_width, img_height, 0),
                aspect="auto",
            )
        else:
            logger.info("No background image provided.")
    else:
        ax.add_patch(
            Rectangle(
                (frame[0][0], frame[1][0]),
                frame[0][1] - frame[0][0],
                frame[1][1] - frame[1][0],
                facecolor="none",
                edgecolor=(0, 0, 0, 0.8),
                linewidth=2,
                zorder=1,
            )
        )

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    show_on_the_fly = (
        kwargs["show"]
        if "show" in kwargs
        else getattr(file_config, "show_on_the_fly", True)
    )
    return finalize_plot(fig, show_on_the_fly)
