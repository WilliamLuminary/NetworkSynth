from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

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
    """Points to pixels at *dpi*, floored at *minimum*.  A point is 1/72 inch."""
    if points is None:
        return minimum
    return max(minimum, round(points * dpi / 72))


def recommend_dpi_cv2(num_nodes: int) -> int:
    """Pick DPI for the CV2/OpenCV renderer (+300 over matplotlib tiers).

    Returns one of 450, 600, 900, 1200, 1500, or 2100.
    """
    return recommend_dpi(num_nodes) + _DPI_BOOST


_WEBP_MAX_PX = 16383


#: Segments per ``cv2.polylines`` call.  A multi-million-edge network is drawn
#: a batch at a time rather than as one array of every segment.
_SEGMENT_BATCH = 1_000_000
_EDGE_COLOR_BGR = (0, 0, 255)
_NODE_COLOR_BGR = (255, 0, 0)


class _CvCanvas(NamedTuple):
    """A blank canvas and the projection onto it.

    Shared by the two OpenCV renderers — the whole-network one and the hybrid
    snapshot — which agree on every pixel and differ only in where their extent
    comes from and what they do with the result.
    """

    image: ndarray
    x_min: float
    y_max: float
    sx: float
    sy: float
    dpi: int

    def project(self, xs, ys):
        """Data coordinates to pixels.  y flips: a plot counts up, an image down."""
        return (
            ((xs - self.x_min) * self.sx).astype(np.int32),
            ((self.y_max - ys) * self.sy).astype(np.int32),
        )


def _cv2_canvas(extent, style, node_count: int) -> _CvCanvas:
    """The canvas to draw *extent* on, at the size *style* asks for.

    *extent* is ``((x_min, x_max), (y_min, y_max))`` before margins: this
    widens it by ``style.margin_frac`` of each span.  Because both margins are
    a fraction of their own span, widening leaves the aspect ratio alone.
    """
    dpi = style.dpi if style.dpi is not None else recommend_dpi_cv2(node_count)

    (x_min, x_max), (y_min, y_max) = extent
    mx = (x_max - x_min) * style.margin_frac
    my = (y_max - y_min) * style.margin_frac
    x_min, x_max = x_min - mx, x_max + mx
    y_min, y_max = y_min - my, y_max + my

    # Guard against a degenerate extent — every node at one position, or a
    # frame with no height: the aspect and both scales divide by these.
    frame_w = max(x_max - x_min, 1e-12)
    frame_h = max(y_max - y_min, 1e-12)

    # Max pixel size of the render, always capped at the WebP 16383px hard
    # limit. A large network at that limit produces a ~190MP image most viewers
    # cannot open, so callers can pass a lower max_px.
    max_px = min(style.max_px or _WEBP_MAX_PX, _WEBP_MAX_PX)
    img_h = int(12 * dpi)
    img_w = int(12 * dpi * frame_w / frame_h)
    # Clamp to max_px while preserving aspect: when either axis exceeds the
    # cap, scale both by the same factor (clamping each axis independently
    # would square a rectangular network).
    clamp = min(1.0, max_px / max(img_h, img_w))
    img_h = max(1, int(img_h * clamp))
    img_w = max(1, int(img_w * clamp))

    return _CvCanvas(
        image=np.full((img_h, img_w, 3), 255, dtype=np.uint8),
        x_min=x_min,
        y_max=y_max,
        sx=(img_w - 1) / frame_w,
        sy=(img_h - 1) / frame_h,
        dpi=dpi,
    )


def _draw_nodes(canvas: _CvCanvas, px, py, style) -> None:
    """Every node inside the canvas as a blue dot."""
    import cv2

    img_h, img_w = canvas.image.shape[:2]
    valid = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    radius = _points_to_px(style.node_size, canvas.dpi, minimum=0) // 2
    if radius < 1:
        # A single pixel per node: cheaper than a circle of radius 0, and
        # the same result.
        canvas.image[py[valid], px[valid]] = np.array(_NODE_COLOR_BGR, dtype=np.uint8)
    else:
        for x, y in zip(px[valid], py[valid]):
            cv2.circle(canvas.image, (int(x), int(y)), radius, _NODE_COLOR_BGR, -1)


def render_network(graph: SynthGraph, style):
    """Render a SynthGraph to a BGR ndarray using OpenCV.

    Only allocates a fixed-size pixel buffer (H × W × 3 bytes) regardless
    of the number of nodes/edges, avoiding OOM on multi-million-element
    graphs.

    *style* supplies dpi, max_px, margin_frac, border and the node/edge sizes.
    """
    import cv2

    pos_arr = graph.positions()
    x_min, y_min = pos_arr.min(axis=0)
    x_max, y_max = pos_arr.max(axis=0)
    canvas = _cv2_canvas(
        ((x_min, x_max), (y_min, y_max)), style, graph.number_of_nodes()
    )
    px, py = canvas.project(pos_arr[:, 0], pos_arr[:, 1])

    # Segments are built per batch from indices into the projected nodes, not
    # once for the whole graph: an array of every segment is what runs out of
    # memory on a network this renderer exists to handle.
    thickness = _points_to_px(style.line_width, canvas.dpi, minimum=1)
    edge_list = np.asarray(list(graph.edges()), dtype=np.int64)
    for start in range(0, len(edge_list), _SEGMENT_BATCH):
        batch = edge_list[start : start + _SEGMENT_BATCH]
        pts_u = np.column_stack([px[batch[:, 0]], py[batch[:, 0]]])
        pts_v = np.column_stack([px[batch[:, 1]], py[batch[:, 1]]])
        segments = np.stack([pts_u, pts_v], axis=1).astype(np.int32)
        cv2.polylines(
            canvas.image,
            segments,
            isClosed=False,
            color=_EDGE_COLOR_BGR,
            thickness=thickness,
        )
    del edge_list

    _draw_nodes(canvas, px, py, style)

    if style.border:
        img_h, img_w = canvas.image.shape[:2]
        cv2.rectangle(canvas.image, (0, 0), (img_w - 1, img_h - 1), (0, 0, 0), 2)

    return canvas.image


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
            f"WebP resize: {w}x{h} -> {new_w}x{new_h} (limit {_WEBP_MAX_PX}px)"
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
    style,
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
    style : RenderStyle
        Supplies dpi and the node/edge sizes, in points.
    """
    dpi = style.dpi
    node_size = style.node_size
    line_width = style.line_width
    assert (
        node_size is not None and line_width is not None
    ), "bfs_snapshot needs node_size and line_width"

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
    style,
) -> None:
    """Render one hybrid Phase 2 snapshot with OpenCV.

    Drawn on the whole *frame* rather than on what has grown so far, so a
    sequence of snapshots is a sequence of the same view.  Sizes are in points,
    converted to pixels at *style.dpi*; unset draws a 1px line and a
    single-pixel node.
    """
    import os

    import cv2

    canvas = _cv2_canvas(frame, style, len(node_positions))

    if edges:
        edge_arr = np.asarray(
            [((e[0][0], e[0][1]), (e[1][0], e[1][1])) for e in edges],
            dtype=np.float64,
        )
        pts_u = np.column_stack(canvas.project(edge_arr[:, 0, 0], edge_arr[:, 0, 1]))
        pts_v = np.column_stack(canvas.project(edge_arr[:, 1, 0], edge_arr[:, 1, 1]))
        segments = np.stack([pts_u, pts_v], axis=1).astype(np.int32)
        thickness = _points_to_px(style.line_width, canvas.dpi, minimum=1)
        for start in range(0, len(segments), _SEGMENT_BATCH):
            cv2.polylines(
                canvas.image,
                segments[start : start + _SEGMENT_BATCH],
                isClosed=False,
                color=_EDGE_COLOR_BGR,
                thickness=thickness,
            )

    if node_positions:
        pos_arr = np.asarray(node_positions, dtype=np.float64)
        _draw_nodes(canvas, *canvas.project(pos_arr[:, 0], pos_arr[:, 1]), style)

    path = os.path.join(output_dir, f"snapshot_{index:05d}.png")
    cv2.imwrite(path, canvas.image)


def plot_network(
    data_type: str,
    graph: SynthGraph,
    style,
    frame_size=None,
    synthetic_frame_size=None,
    background=None,
):
    """Plot a network with matplotlib, styled by ``config.render(data_type)``."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    node_size = style.node_size
    line_width = style.line_width
    assert (
        node_size is not None and line_width is not None
    ), f"{data_type} needs node_size and line_width"

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

    fig = Figure(figsize=(fig_width, fig_height), dpi=style.dpi)
    ax = fig.add_subplot(111)

    positions = graph.positions()
    edge_width = line_width
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

    for node in graph.nodes():
        pos = positions[node]
        ax.plot(pos[0], pos[1], "bo", markersize=node_size, zorder=2)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type == "original_graph":
        image = background
        if image is not None:
            alpha = style.alpha
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

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    return finalize_plot(fig, style.show_on_the_fly)
