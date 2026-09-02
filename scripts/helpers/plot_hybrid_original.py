# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "input", "samples", "hybrid_mode"
)


def load_sample_network(sample: str):
    set_name = f"sample_{sample}"

    pos_path = os.path.join(BASE_INPUT_PATH, f"{set_name}_pos.npy")
    mat_path = os.path.join(BASE_INPUT_PATH, f"{set_name}_mat.npy")
    if not os.path.isfile(pos_path):
        sys.exit(f"Positions file not found: {pos_path}")
    if not os.path.isfile(mat_path):
        sys.exit(f"Adjacency matrix file not found: {mat_path}")

    positions = np.load(pos_path, allow_pickle=True)
    mat = np.load(mat_path, allow_pickle=True).item()

    from networksynth.graphs.synth_graph import SynthGraph

    graph = SynthGraph.from_sparse_matrix(positions, mat).largest_connected_component()

    pos = graph.positions()
    pos[:, [0, 1]] = pos[:, [1, 0]]

    logger.info(
        "Loaded sample_%s: %d nodes, %d edges",
        sample,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def load_sample_image(sample: str):
    set_name = f"sample_{sample}"
    img_path = os.path.join(BASE_INPUT_PATH, f"{set_name}_image.tif")
    if not os.path.isfile(img_path):
        logger.warning("No background image found: %s", img_path)
        return None
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", img_path)
        return None

    frame_size = (510, 510)
    h, w = image.shape[:2]
    scale = max(frame_size) / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image


def plot_original_graph(graph, background=None, title=None):
    from matplotlib.figure import Figure

    frame_size = (510, 510)
    frame = ((0, frame_size[0]), (0, frame_size[1]))

    frame_width = frame[0][1] - frame[0][0]
    frame_height = frame[1][1] - frame[1][0]
    aspect_ratio = frame_width / frame_height
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig = Figure(figsize=(fig_width, fig_height), dpi=300)
    ax = fig.add_subplot(111)

    positions = graph.positions()
    edge_width = 3.0
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

    node_size = 6.0
    for node in graph.nodes():
        pos = positions[node]
        ax.plot(pos[0], pos[1], "bo", markersize=node_size, zorder=2)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if background is not None:
        alpha = 0.6
        img_height, img_width = background.shape[:2]
        ax.imshow(
            background,
            cmap="gray",
            alpha=alpha,
            extent=(0, img_width, img_height, 0),
            aspect="auto",
        )

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


def main(
    sample: str = "A",
    output: str = None,
    fmt: str = "png",
    no_background: bool = False,
):
    """Plot the original network of a hybrid-mode sample.

    Args:
        sample: Sample letter — A, B, C, or D.
        output: Output file path. Default: sample_<X>_original_graph.<fmt>
        fmt: Output format (png, svg, pdf, webp).
        no_background: If set, omit the background image.
    """
    sample = sample.upper()
    if sample not in ("A", "B", "C", "D"):
        sys.exit(f"Invalid sample: {sample}. Use A, B, C, or D.")

    graph = load_sample_network(sample)
    background = None if no_background else load_sample_image(sample)

    fig = plot_original_graph(
        graph,
        background=background,
        title=f"Sample {sample} — Original Network",
    )

    if output is None:
        output = f"sample_{sample}_original_graph.{fmt}"

    if fmt == "webp":
        from networksynth.utils import save_figure_as_webp

        try:
            save_figure_as_webp(fig, output, dpi=300)
        except Exception as e:
            logger.warning("WebP failed (%s), falling back to PNG", e)
            output = output.rsplit(".", 1)[0] + ".png"
            fig.savefig(output, format="png", bbox_inches="tight", dpi=300)
    else:
        save_kwargs = {"format": fmt, "bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(output, **save_kwargs)

    from matplotlib import pyplot as plt

    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
