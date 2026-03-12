# src/configs/file_definitions.py
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .enums import DataType

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SaveSpec + serialisation functions
# ---------------------------------------------------------------------------


@dataclass
class SaveSpec:
    """Describes how a DataType should be persisted.

    Attributes
    ----------
    relative_dir : str
        Sub-directory under the dataset output dir (e.g. "original", "synthetic", "").
    detail : str
        Descriptive tag embedded in the file name (e.g. "synthetic_edgelist").
    save_fn : Callable[[Any, str], None]
        ``(content, filepath) -> None`` — the actual serialisation function.
    use_timestamp : bool
        Whether to append a timestamp to the file name (default ``True``).
    """

    relative_dir: str
    detail: str
    save_fn: Callable[[Any, str], None]
    use_timestamp: bool = True


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


def make_generate_save_specs():
    """Factory that returns the standard SaveSpecs for generate mode.

    Modes that extend generate (mosaic, scaling, hybrid, attr-generate)
    can call this and merge additional specs via ``{**make_generate_save_specs(), ...}``.
    """
    return {
        DataType.ORIGINAL_IMAGE: SaveSpec("original", "original_image", save_png),
        DataType.ORIGINAL_GRAPH: SaveSpec("original", "original_graph", save_svg),
        DataType.ORIGINAL_NETWORK: SaveSpec(
            "original", "original_network", save_pickle
        ),
        DataType.ORIGINAL_PROPERTY: SaveSpec(
            "original", "original_property", save_pickle
        ),
        DataType.SYNTHETIC_GRAPH: SaveSpec("synthetic", "synthetic_graph", save_webp),
        DataType.SYNTHETIC_NETWORK: SaveSpec(
            "synthetic", "synthetic_network", save_pickle
        ),
        DataType.SYNTHETIC_EDGELIST: SaveSpec(
            "synthetic", "synthetic_edgelist", save_csv
        ),
        DataType.SYNTHETIC_POSITIONS: SaveSpec(
            "synthetic", "synthetic_positions", save_csv
        ),
        DataType.SYNTHETIC_NETWORK_NKI: SaveSpec(
            "synthetic", "synthetic_network_nki", save_networkit
        ),
        DataType.ANALYSIS_DATA: SaveSpec("", "analysis_data", save_pickle),
        DataType.ANALYSIS_FIGURE: SaveSpec("", "analysis_figure", save_svg),
    }


# ---------------------------------------------------------------------------
# Plot / image display configuration (used by RunAgent.plot_network,
# NOT by Saver).  Kept for rendering parameters only.
# ---------------------------------------------------------------------------


@dataclass
class FileConfig:
    relative_dir: str
    data_type: DataType
    detail: Optional[str] = None

    def __post_init__(self):
        self.file_tags = self.data_type.tags
        self.file_extension = self.data_type.file_extension


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
        data_type=DataType.ORIGINAL_IMAGE,
        alpha=0.6,
        detail="original_image",
    ),
    DataType.ORIGINAL_GRAPH: PlotConfig(
        relative_dir=ORIGINAL_DIR,
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.ORIGINAL_GRAPH,
        detail="original_graph",
    ),
    DataType.ORIGINAL_NETWORK: FileConfig(
        relative_dir=ORIGINAL_DIR,
        data_type=DataType.ORIGINAL_NETWORK,
        detail="original_network",
    ),
    DataType.ORIGINAL_PROPERTY: FileConfig(
        relative_dir=ORIGINAL_DIR,
        data_type=DataType.ORIGINAL_PROPERTY,
        detail="original_property",
    ),
    DataType.SYNTHETIC_GRAPH: PlotConfig(
        relative_dir=SYNTHETIC_DIR,
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.SYNTHETIC_GRAPH,
        show_on_the_fly=False,
        detail="synthetic_graph",
    ),
    DataType.SYNTHETIC_GRAPH_PNG: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_GRAPH_PNG,
        detail="synthetic_graph_png",
    ),
    DataType.SYNTHETIC_NETWORK: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_NETWORK,
        detail="synthetic_network",
    ),
    DataType.SYNTHETIC_EDGELIST: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_EDGELIST,
        detail="synthetic_edgelist",
    ),
    DataType.SYNTHETIC_POSITIONS: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_POSITIONS,
        detail="synthetic_positions",
    ),
    DataType.SYNTHETIC_NETWORK_NKI: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_NETWORK_NKI,
        detail="synthetic_network_nki",
    ),
    DataType.ANALYSIS_DATA: FileConfig(
        relative_dir=INPLACE_DIR,
        data_type=DataType.ANALYSIS_DATA,
        detail="analysis_data",
    ),
    DataType.ANALYSIS_FIGURE: PlotConfig(
        relative_dir=INPLACE_DIR,
        data_type=DataType.ANALYSIS_FIGURE,
        detail="analysis_figure",
    ),
    DataType.DEFAULT_DATA: FileConfig(
        relative_dir="",
        data_type=DataType.DEFAULT_DATA,
    ),
}
