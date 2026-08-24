import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation functions (Layer 1)
# ---------------------------------------------------------------------------


def save_pickle(obj: Any, filepath: str) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def save_csv(content, filepath: str) -> None:
    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in content:
            writer.writerow(row)


def save_networkit(content, filepath: str) -> None:
    """Save a networkit graph in binary format.

    Accepts either a bare ``nk.Graph`` or a ``(nk.Graph, positions)``
    tuple.  When positions are provided, a companion
    ``<basename>_positions.npy`` is written alongside the ``.nkbin``
    so that the nkbin output is self-contained (topology + weights
    in the binary graph, positions in the compact numpy array).
    """
    import networkit as nk
    import numpy as np

    if isinstance(content, tuple):
        nk_graph, positions = content
    else:
        nk_graph = content
        positions = None

    nk.writeGraph(nk_graph, filepath, nk.Format.NetworkitBinary)

    if positions is not None:
        pos_path = filepath.rsplit(".", 1)[0] + "_positions.npy"
        np.save(pos_path, np.asarray(positions))
        _logger.info(f"Saved companion positions: {pos_path}")


def save_webp(fig_or_image, filepath: str) -> None:
    from numpy import ndarray
    from PIL import Image as _Image

    if isinstance(fig_or_image, _Image.Image):
        fig_or_image.save(filepath, "webp", lossless=True)
        return

    if isinstance(fig_or_image, ndarray):
        arr = fig_or_image
        if arr.ndim == 3 and arr.shape[2] == 3:
            import cv2

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        _Image.fromarray(arr).save(filepath, "webp", lossless=True)
        return

    from matplotlib.figure import Figure

    assert isinstance(
        fig_or_image, Figure
    ), f"WebP saving expects a matplotlib Figure or PIL Image, got {type(fig_or_image).__name__}."
    from utils import save_figure_as_webp

    save_figure_as_webp(fig_or_image, filepath)

    from matplotlib import pyplot as _plt

    _plt.close(fig_or_image)


def save_svg(fig, filepath: str) -> None:
    from matplotlib.figure import Figure

    assert isinstance(
        fig, Figure
    ), f"SVG saving expects a matplotlib Figure, got {type(fig).__name__}."
    fig.savefig(filepath, format="svg", bbox_inches="tight")

    from matplotlib import pyplot as _plt

    _plt.close(fig)


def save_png(image, filepath: str) -> None:
    from matplotlib.figure import Figure
    from numpy import ndarray
    from PIL import Image as _Image

    if isinstance(image, _Image.Image):
        image.save(filepath, "png")
    elif isinstance(image, Figure):
        image.savefig(filepath, format="png", bbox_inches="tight")

        from matplotlib import pyplot as _plt

        _plt.close(image)
    elif isinstance(image, ndarray):
        import cv2

        cv2.imwrite(filepath, image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def save_text(text: str, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(text)


def save_network_csv(graph, filepath: str) -> None:
    """Save a SynthGraph as two CSVs: edgelist and positions.

    Given ``filepath`` (e.g. ``…/synthetic_network.csv``), writes:
    - ``…/synthetic_network_edgelist.csv``
    - ``…/synthetic_network_positions.csv``

    The weight column is written only for a weighted graph: the reader takes
    its presence as the answer, and an unweighted networkit graph reports every
    weight as 1.0.
    """
    base, ext = os.path.splitext(filepath)

    if graph.is_weighted():
        edgelist_rows = [["source_index", "target_index", "edge_weight"]]
        edgelist_rows.extend([u, v, w] for u, v, w in graph.edges_with_weights())
    else:
        edgelist_rows = [["source_index", "target_index"]]
        edgelist_rows.extend([u, v] for u, v in graph.edges())
    save_csv(edgelist_rows, f"{base}_edgelist{ext}")

    positions = graph.positions()
    position_rows = [["x", "y"]]
    for pos in positions:
        position_rows.append([pos[0], pos[1]])
    save_csv(position_rows, f"{base}_positions{ext}")


def save_network_nkbin(graph, filepath: str) -> None:
    save_networkit((graph.nk, graph.positions()), filepath)


# Rendering styles.  Nothing here concerns where a file is written.


@dataclass(frozen=True)
class RenderStyle:
    """How one output is drawn.  Frozen, so a config cannot restyle another's."""

    #: Points, matplotlib's unit for markersize/linewidth.  The OpenCV
    #: renderers convert to pixels at dpi.  None = thinnest possible.
    node_size: Optional[float] = None
    line_width: Optional[float] = None
    #: Opacity of a background image, for outputs that have one.
    alpha: float = 0.6
    #: None lets the renderer choose from the node count.
    dpi: Optional[int] = None
    #: Cap on the longest side in pixels.
    max_px: Optional[int] = None
    margin_frac: float = 0.02
    border: bool = False
    show_on_the_fly: bool = True


# Per-output defaults live on BaseConfig as RENDER_* / SAVE_* attributes.


ORIGINAL_DIR = "original"
SYNTHETIC_DIR = "synthetic"
INPLACE_DIR = ""

# Save specs.


@dataclass(frozen=True)
class SaveSpec:
    """One file to write for an output.

    Keyword-named so a spec reads as what it is::

        SaveSpec(ORIGINAL_DIR, "report", "txt", save_text, use_timestamp=False)
    """

    #: Subdirectory under the run's output root ("" = the root).
    relative_dir: str
    detail: str
    #: Without the dot.
    extension: str
    #: Called as save_fn(content, filepath).
    save_fn: Callable[[Any, str], None]
    use_timestamp: bool = True
