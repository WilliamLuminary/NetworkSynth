# src/configs/file_definitions.py
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from .enums import DataType

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


def save_webp(fig, filepath: str) -> None:
    from matplotlib.figure import Figure

    assert isinstance(
        fig, Figure
    ), f"WebP saving expects a matplotlib Figure, got {type(fig).__name__}."
    from utils import save_figure_as_webp

    save_figure_as_webp(fig, filepath)

    from matplotlib import pyplot as _plt

    _plt.close(fig)


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

    if isinstance(image, Figure):
        image.savefig(filepath, format="png", bbox_inches="tight")
    elif isinstance(image, ndarray):
        import cv2

        cv2.imwrite(filepath, image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


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
    """Extract networkit graph + positions and save in binary format."""
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

FILE_CONFIGURATIONS = {
    DataType.ORIGINAL_IMAGE: ImageConfig(
        relative_dir=ORIGINAL_DIR,
        alpha=0.6,
        detail="original_image",
    ),
    DataType.ORIGINAL_GRAPH: PlotConfig(
        relative_dir=ORIGINAL_DIR,
        node_size=6.0,
        line_width=3.0,
        detail="original_graph",
    ),
    DataType.ORIGINAL_NETWORK: FileConfig(
        relative_dir=ORIGINAL_DIR,
        detail="original_network",
    ),
    DataType.ORIGINAL_PROPERTY: FileConfig(
        relative_dir=ORIGINAL_DIR,
        detail="original_property",
    ),
    DataType.SYNTHETIC_GRAPH: PlotConfig(
        relative_dir=SYNTHETIC_DIR,
        node_size=6.0,
        line_width=3.0,
        show_on_the_fly=False,
        detail="synthetic_graph",
    ),
    DataType.ANALYSIS_FIGURE: PlotConfig(
        relative_dir=INPLACE_DIR,
        detail="analysis_figure",
    ),
}
