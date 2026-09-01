# SPDX-License-Identifier: GPL-3.0-or-later
import json
import logging
import os
import re
import sys
from dataclasses import replace

from configs.gui_config import GuiConfig
from handlers import AttributesCalculator, configure_console
from utils import plot_network

logger = logging.getLogger("gui_preview")

PREVIEW_DPI = 90

_EXPORT_SUFFIX = ".graphml.gz"
_EXPORT_MARKER = "synthetic_network"

# One file per network, numbered by this infix. There is no combined batch any
# more: holding a list took a pickle, and GraphML carries a single graph.
_EXPORT_INFIX = re.compile(r"_n(\d+)_" + _EXPORT_MARKER)

_LABEL_WIDTH = 21


def _row(label: str, value: str) -> str:
    return f"{label + ':':<{_LABEL_WIDTH}}{value}"


def _info_text(title: str, graph, attributes) -> str:
    rows = [
        _row("Nodes", f"{graph.number_of_nodes():,}"),
        _row("Edges", f"{graph.number_of_edges():,}"),
        _row("Average degree", f"{attributes.average_degree:.4f}"),
        _row("Average edge length", f"{attributes.average_length:.4f}"),
    ]
    lines = [title, "=" * max(len(row) for row in [title] + rows), *rows]
    if attributes.degree_distribution:
        lines.append("")
        lines.append("Degree distribution:")
        for degree in sorted(attributes.degree_distribution):
            fraction = attributes.degree_distribution[degree]
            lines.append(f"  degree {degree:>3d}: {fraction:.4f}")
    return "\n".join(lines)


def _write(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, transparent=False)
    return path


def _preview_style(config, identifier: str):
    return replace(config.render(identifier), dpi=PREVIEW_DPI, show_on_the_fly=False)


def _background_figure(image, frame_size, style):
    from matplotlib.figure import Figure

    width, height = frame_size
    fig = Figure(figsize=(10 * width / height, 10), dpi=style.dpi)
    ax = fig.add_subplot(111)

    img_height, img_width = image.shape[:2]
    ax.imshow(
        image,
        cmap="gray",
        alpha=style.alpha,
        extent=(0, img_width, img_height, 0),
        aspect="auto",
    )
    ax.set_xlim((0, width))
    ax.set_ylim((0, height))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def preview_original(config, out_dir: str) -> dict:
    datasets = config.get_datasets()
    dataset_id = datasets[0]

    graph = config.ORIGINAL_NETWORK_FUNC(dataset_id)
    image = config.ORIGINAL_IMAGE_FUNC(dataset_id)
    attributes = AttributesCalculator().analyze(graph)
    style = _preview_style(config, "original_graph")

    images = {
        "network": _write(
            plot_network(
                data_type="original_graph",
                graph=graph,
                style=style,
                frame_size=config.FRAME_SIZE,
            ),
            out_dir,
            "network.png",
        )
    }
    if image is not None:
        images["both"] = _write(
            plot_network(
                data_type="original_graph",
                graph=graph,
                style=style,
                frame_size=config.FRAME_SIZE,
                background=image,
            ),
            out_dir,
            "both.png",
        )
        images["background"] = _write(
            _background_figure(image, config.FRAME_SIZE, style),
            out_dir,
            "background.png",
        )

    note = ""
    if len(datasets) > 1:
        note = (
            f"{dataset_id} — the first of {len(datasets)} networks in that directory."
        )

    return {
        "kind": "original",
        "has_background": image is not None,
        "image_size": None if image is None else list(reversed(config.IMAGE_SIZE)),
        "images": images,
        "note": note,
        "text": _info_text("Original network", graph, attributes),
    }


def _find_exports(run_root: str) -> list:
    """Every synthetic network the run wrote, in the order it generated them."""
    found = []
    for directory, _, names in os.walk(run_root):
        for name in names:
            match = _EXPORT_INFIX.search(name)
            if match and name.endswith(_EXPORT_SUFFIX):
                found.append((int(match.group(1)), os.path.join(directory, name)))
    if not found:
        raise FileNotFoundError(
            f"no '*{_EXPORT_MARKER}*{_EXPORT_SUFFIX}' under {run_root}. Only "
            "generate mode writes the synthetic networks a preview reads."
        )
    return [path for _, path in sorted(found)]


def preview_synthetic(config, run_root: str, out_dir: str) -> dict:
    from graphs.graphml_io import read_graph_graphml

    exports = _find_exports(run_root)
    graph = read_graph_graphml(exports[0])
    attributes = AttributesCalculator().analyze(graph)
    figure = plot_network(
        data_type="synthetic_graph",
        graph=graph,
        style=_preview_style(config, "synthetic_graph"),
        synthetic_frame_size=config.SYNTHETIC_FRAME_SIZE,
    )

    return {
        "kind": "synthetic",
        "has_background": False,
        "images": {"network": _write(figure, out_dir, "synthetic.png")},
        "count": len(exports),
        "note": (
            f"The first of the {len(exports)} networks in this run's output "
            "list — the order they were generated in, not a ranking."
        ),
        "text": _info_text(
            "Synthetic network — first in the output list", graph, attributes
        ),
    }


def _source(config, dataset_id):
    paths = config.PATHS
    if paths.get("datasets_dir"):
        return paths["datasets_dir"], str(dataset_id)
    for key, suffix in (
        ("edge_list", "_edgelist.csv"),
        ("adjacency", "_adjacency.npy"),
        ("adjacency", "_mat.npy"),
        ("network_graphml", ".graphml.gz"),
        ("network_graphml", ".graphml"),
    ):
        if paths.get(key):
            name = os.path.basename(paths[key])
            if name.endswith(suffix):
                name = name[: -len(suffix)]
            return os.path.dirname(paths[key]), name
    raise ValueError("the spec names no input to edit")


def save_edited(config, out_dir: str) -> dict:
    import shutil

    from configs.file_definitions import save_network_csv

    datasets = config.get_datasets()
    source_dir, _ = _source(config, datasets[0])
    target_dir = os.path.join(source_dir, "edited")
    os.makedirs(target_dir, exist_ok=True)

    written = []
    for dataset_id in datasets:
        _, name = _source(config, dataset_id)
        base = f"{name}_edited"
        graph = config.ORIGINAL_NETWORK_FUNC(dataset_id)
        save_network_csv(graph, os.path.join(target_dir, f"{base}.csv"))
        written.append(base)

        image_path = _image_path(config, dataset_id)
        if image_path:
            shutil.copyfile(image_path, os.path.join(target_dir, f"{base}_image.tif"))

    logger.info(f"Wrote {len(written)} edited network(s) to {target_dir}")
    first = os.path.join(target_dir, f"{written[0]}")
    return {
        "kind": "save_edited",
        "dir": target_dir,
        "count": len(written),
        "inputs": (
            {"datasets_dir": target_dir}
            if config.PATHS.get("datasets_dir")
            else {
                "edge_list": f"{first}_edgelist.csv",
                "positions": f"{first}_positions.csv",
            }
        ),
    }


def _image_path(config, dataset_id):
    directory = config.PATHS.get("datasets_dir")
    if directory:
        candidate = os.path.join(directory, f"{dataset_id}_image.tif")
        return candidate if os.path.exists(candidate) else None
    path = config.PATHS.get("image")
    return path if path and os.path.exists(path) else None


def main(argv=None) -> int:
    configure_console()
    argv = sys.argv if argv is None else argv
    if len(argv) < 4:
        raise SystemExit(
            "usage: gui_preview.py <run_spec.json> original|synthetic "
            "<out_dir> [run_root]"
        )

    spec_path, kind, out_dir = argv[1], argv[2], argv[3]
    os.makedirs(out_dir, exist_ok=True)

    config = GuiConfig.from_spec(spec_path)
    config.initialize()

    if kind == "original":
        info = preview_original(config, out_dir)
    elif kind == "synthetic":
        if len(argv) < 5:
            raise SystemExit("a synthetic preview needs the run's output root")
        info = preview_synthetic(config, argv[4], out_dir)
    elif kind == "save_edited":
        info = save_edited(config, out_dir)
    else:
        raise SystemExit(f"unknown preview kind {kind!r}")

    with open(os.path.join(out_dir, "info.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    logger.info(f"Preview written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
