# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)


def save_pickle(obj: Any, filepath: str) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj: Any, filepath: str) -> None:
    import json

    def plain(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"cannot serialise {type(value).__name__}")

    with open(filepath, "w") as handle:
        json.dump(obj, handle, indent=2, default=plain, sort_keys=True)


def save_csv(content, filepath: str) -> None:
    import csv

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in content:
            writer.writerow(row)


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

    from utils import save_figure_as_webp

    save_figure_as_webp(fig_or_image, filepath)

    from matplotlib import pyplot as _plt

    _plt.close(fig_or_image)


def save_svg(fig, filepath: str) -> None:
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


def save_network_graphml(graph, filepath: str) -> None:
    from graphs.graphml_io import write_graph_graphml

    write_graph_graphml(graph, filepath)


@dataclass(frozen=True)
class RenderStyle:

    node_size: Optional[float] = None
    line_width: Optional[float] = None
    alpha: float = 0.6
    dpi: Optional[int] = None
    max_px: Optional[int] = None
    margin_frac: float = 0.02
    border: bool = False
    show_on_the_fly: bool = True


ORIGINAL_DIR = "original"
SYNTHETIC_DIR = "synthetic"
INPLACE_DIR = ""


@dataclass(frozen=True)
class SaveSpec:

    relative_dir: str
    detail: str
    extension: str
    save_fn: Callable[[Any, str], None]
    use_timestamp: bool = True
