import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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
    """
    base, ext = os.path.splitext(filepath)

    edgelist_rows = [["source_index", "target_index", "edge_weight"]]
    for u, v, w in graph.edges_with_weights():
        edgelist_rows.append([u, v, w])
    save_csv(edgelist_rows, f"{base}_edgelist{ext}")

    positions = graph.positions()
    position_rows = [["x", "y"]]
    for pos in positions:
        position_rows.append([pos[0], pos[1]])
    save_csv(position_rows, f"{base}_positions{ext}")


def save_network_nkbin(graph, filepath: str) -> None:
    save_networkit((graph.nk, graph.positions()), filepath)


# ---------------------------------------------------------------------------
# Plot / image display configuration (used by RunAgent.plot_network,
# NOT by Saver).  Kept for rendering parameters only.
# ---------------------------------------------------------------------------


@dataclass
class FileConfig:
    relative_dir: str
    detail: Optional[str] = None


@dataclass
class ImageConfig(FileConfig):
    alpha: Optional[float] = 0.6


@dataclass
class PlotConfig(FileConfig):
    node_size: float = 6.0
    line_width: float = 3.0
    show_on_the_fly: bool = True


ORIGINAL_DIR = "original"
SYNTHETIC_DIR = "synthetic"
INPLACE_DIR = ""

# Default save specifications, keyed by identifier string.
#
# Usage:
#   Pipeline code calls  saver.save(content, "<identifier>", prefix)
#   which resolves to    BaseConfig.save("<identifier>")
#   which returns the spec list defined here (unless a mode config
#   overrides it — see below).
#
# Each spec is a tuple of 4 or 5 elements:
#   (relative_dir, detail, extension, save_fn[, use_timestamp])
#
#   relative_dir   – subdirectory under the output root (e.g. "original")
#   detail         – descriptive stem used in the filename
#   extension      – file extension without the dot (e.g. "csv", "pkl")
#   save_fn        – serialiser function with signature (content, filepath)
#   use_timestamp  – optional; defaults to True.  Set to False for files
#                    that should never carry a timestamp (e.g. analysis_data).
#
# To override from a mode config, define a classmethod on the config:
#
#   class MyConfig(BaseConfig):
#       @classmethod
#       def save_original_network(cls):
#           return [("original", "original_network", "csv", save_network_csv)]
#
# The dispatcher checks for a save_<identifier> method first; if none
# exists, it falls back to this dict.
# Default formats: images -> webp, network exports -> csv.
# The save_svg (vector image) and save_network_nkbin (.nkbin + companion
# .npy positions) serializers remain available and can be re-enabled per
# config by defining a save_<identifier>() classmethod.
DEFAULT_SAVE_SPECS = {
    "original_image": [("original", "original_image", "webp", save_webp)],
    "original_network": [
        ("original", "original_network", "csv", save_network_csv),
    ],
    "original_property": [("original", "original_property", "pkl", save_pickle)],
    "original_report": [("original", "report", "txt", save_text, False)],
    "original_graph": [("original", "original_graph", "webp", save_webp)],
    "synthetic_graph": [("synthetic", "synthetic_graph", "webp", save_webp)],
    "synthetic_report": [("synthetic", "report", "txt", save_text, False)],
    "synthetic_network": [("synthetic", "synthetic_network", "pkl", save_pickle)],
    "synthetic_export": [
        ("synthetic", "synthetic_network", "csv", save_network_csv),
    ],
    "analysis_data": [("", "analysis_data", "pkl", save_pickle, False)],
    "analysis_figure": [("", "analysis_figure", "webp", save_webp)],
}

FILE_CONFIGURATIONS = {
    "original_image": ImageConfig(
        relative_dir=ORIGINAL_DIR,
        alpha=0.6,
        detail="original_image",
    ),
    "original_graph": PlotConfig(
        relative_dir=ORIGINAL_DIR,
        node_size=6.0,
        line_width=3.0,
        detail="original_graph",
    ),
    "original_network": FileConfig(
        relative_dir=ORIGINAL_DIR,
        detail="original_network",
    ),
    "original_property": FileConfig(
        relative_dir=ORIGINAL_DIR,
        detail="original_property",
    ),
    "synthetic_graph": PlotConfig(
        relative_dir=SYNTHETIC_DIR,
        node_size=6.0,
        line_width=3.0,
        show_on_the_fly=False,
        detail="synthetic_graph",
    ),
    "analysis_figure": PlotConfig(
        relative_dir=INPLACE_DIR,
        detail="analysis_figure",
    ),
}
