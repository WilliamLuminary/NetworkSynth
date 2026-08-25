"""Render what a run reads, or what one produced, without running anything.

A separate process for the same reason ``gui_run.py`` is one: loading a
network, measuring it and rendering it takes seconds, and the window must stay
answerable while that happens.

It is handed the run-spec the form has already built, so a preview is drawn by
the same loader, the same style and the same frame as the run's own output —
"background and network" is the figure the run writes as ``original_graph``,
not an imitation of it.  Nothing is written into the run's output directory:
everything lands in the directory named on the command line.

    gui_preview.py <run_spec.json> original    <out_dir>
    gui_preview.py <run_spec.json> synthetic   <out_dir> <run_root>
    gui_preview.py <run_spec.json> save_edited <out_dir>

The last one writes rather than draws: the input network as the spec is
currently reading it, turn included, so a correction made in the form can be
kept.
"""

import json
import logging
import os
import pickle
import sys
from dataclasses import replace

from configs.gui_config import GuiConfig
from handlers import AttributesCalculator, configure_console
from utils import plot_network

logger = logging.getLogger("gui_preview")

#: Previews are shown in a panel a few hundred pixels wide.  Node and line
#: sizes are in points, so a lower DPI is the same picture with fewer pixels —
#: at 90 a render is ~900px and takes about half as long as the 300 the run
#: saves at, nearly all of it in encoding the PNG.  Lower buys little: below
#: this the per-edge cost of building the figure dominates, and it does not
#: change with DPI.
PREVIEW_DPI = 90

#: The batch file ``GenerationRun.save_synthetic_outputs`` writes: every
#: synthetic network from the run, in the order they were collected.
_BATCH_SUFFIX = ".pkl"
_BATCH_MARKER = "synthetic_network"

_LABEL_WIDTH = 21


def _row(label: str, value: str) -> str:
    return f"{label + ':':<{_LABEL_WIDTH}}{value}"


def _info_text(title: str, graph, attributes) -> str:
    lines = [
        title,
        "=" * 40,
        _row("Nodes", f"{graph.number_of_nodes():,}"),
        _row("Edges", f"{graph.number_of_edges():,}"),
        _row("Average degree", f"{attributes.average_degree:.4f}"),
        _row("Average edge length", f"{attributes.average_length:.4f}"),
    ]
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
    """The run's own style, at preview size and without an OpenCV window.

    ``show_on_the_fly`` is on for some outputs, which would pop up a cv2
    window from a process the user cannot see.
    """
    return replace(config.render(identifier), dpi=PREVIEW_DPI, show_on_the_fly=False)


def _background_figure(image, frame_size, style):
    """The background on its own, in the frame the network is drawn in.

    Mirrors ``plot_network``'s geometry for ``original_graph`` so the three
    original previews are one picture with layers removed, not three
    framings of the same data.  ``plot_network`` itself always draws the
    nodes and edges, which is exactly what this variant leaves out.
    """
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
    """The input network: with its background, without it, and the background.

    All three are rendered now rather than one per request, so switching
    between them costs nothing.  Only the first dataset is drawn: a directory
    of networks is many runs' worth of input, and a preview is a look at what
    was chosen, not a contact sheet.
    """
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

    # Named only when the name means something: a directory's datasets are
    # named after their files, where a single CSV pair takes the run's name.
    note = ""
    if len(datasets) > 1:
        note = (
            f"{dataset_id} — the first of {len(datasets)} networks in that directory."
        )

    return {
        "kind": "original",
        "has_background": image is not None,
        # (width, height).  The form takes its frame from this: position data
        # need not reach the corners of what it was traced from, so the image
        # is the only reliable statement of how big the input really is.
        "image_size": None if image is None else [image.shape[1], image.shape[0]],
        "images": images,
        "note": note,
        "text": _info_text("Original network", graph, attributes),
    }


def _find_batch(run_root: str) -> str:
    """The run's synthetic batch file, newest first when a run held several.

    Raises rather than reporting an empty preview: the caller only asks after a
    run finished, so a missing batch means the run did not write one — which is
    true of every mode except ``generate`` — and saying so is more use than an
    empty panel.
    """
    found = []
    for directory, _, names in os.walk(run_root):
        for name in names:
            if _BATCH_MARKER in name and name.endswith(_BATCH_SUFFIX):
                found.append(os.path.join(directory, name))
    if not found:
        raise FileNotFoundError(
            f"no '*{_BATCH_MARKER}*{_BATCH_SUFFIX}' under {run_root}. Only "
            "generate mode writes the batch of synthetic networks a preview "
            "reads."
        )
    return max(found, key=os.path.getmtime)


def preview_synthetic(config, run_root: str, out_dir: str) -> dict:
    """The first network in the run's output list.

    The first, deliberately, and labelled as such: the list is in the order
    the workers finished, so it is not ranked, and calling it the best would
    claim a comparison nobody made.
    """
    batch_path = _find_batch(run_root)
    with open(batch_path, "rb") as handle:
        graphs = pickle.load(handle)
    if not graphs:
        raise ValueError(f"{batch_path} holds no networks")

    graph = graphs[0]
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
        "count": len(graphs),
        "note": (
            f"The first of the {len(graphs)} networks in this run's output "
            "list — the order they were generated in, not a ranking."
        ),
        "text": _info_text(
            "Synthetic network — first in the output list", graph, attributes
        ),
    }


def _source(config, dataset_id):
    """Where the input came from, and what to call an edited copy of it."""
    paths = config.PATHS
    if paths.get("datasets_dir"):
        return paths["datasets_dir"], str(dataset_id)
    for key, suffix in (
        ("edge_list", "_edgelist.csv"),
        ("adjacency", "_mat.npy"),
        ("network_pkl", ".pkl"),
    ):
        if paths.get(key):
            name = os.path.basename(paths[key])
            if name.endswith(suffix):
                name = name[: -len(suffix)]
            return os.path.dirname(paths[key]), name
    raise ValueError("the spec names no input to edit")


def save_edited(config, out_dir: str) -> dict:
    """Write the input network as it is being read, turn and all.

    Into an ``edited`` folder beside the source rather than alongside it: a
    directory of networks is discovered by scanning for ``*_edgelist.csv``, so
    a copy dropped in there would quietly double every later run.

    The image is copied across untouched when there is one, so the result is a
    complete dataset — nothing about the image is edited here, or anywhere.
    """
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
        # What the form should read from now: the edited copies, which already
        # have the turn baked in.
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
    """The image file behind *dataset_id*, if the spec has one."""
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
