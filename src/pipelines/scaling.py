# src/pipelines/scaling.py
"""
Scaling pipeline — multi-root synchronized BFS for large networks.
"""

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from config import BaseConfig, DataType
from config.scaling_mode import ScalingConfig
from graph import GraphGenerator
from graph.synth_graph import SynthGraph
from handlers import RunAgent, Saver
from utils import finalize_plot, trim_graph

ScalingConfig.initialize()

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Main pipeline for one dataset
# ------------------------------------------------------------------ #
def run_scaling_for_dataset(dataset_id):
    """Run the full scaling pipeline for a single dataset."""
    logger.info(f"=== Scaling pipeline for dataset: {dataset_id} ===")
    logger.info(BaseConfig())

    # 1. Load original network → compute structural attributes
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    attributes = data_agent.attributes

    # 2. Generate scaled network
    generator = GraphGenerator(attributes)
    scaled_graph = generator.generate_scaled_network(
        scale_rows=BaseConfig.SCALE_ROWS,
        scale_cols=BaseConfig.SCALE_COLS,
        max_rounds=BaseConfig.MAX_GENERATION_ROUNDS,
        root_spacing_factor=BaseConfig.ROOT_SPACING_FACTOR,
    )

    # 3. Trim to match average degree
    scaled_graph = trim_graph(scaled_graph, attributes.average_degree)

    # 4. Save network data + plot
    data_agent.add_synthetic_graph(scaled_graph)
    prefix = f"scaled_{BaseConfig.SCALE_ROWS}x{BaseConfig.SCALE_COLS}"

    data_agent.save(DataType.SYNTHETIC_EDGELIST, f"{prefix}_", arg=scaled_graph)
    data_agent.save(DataType.SYNTHETIC_POSITIONS, f"{prefix}_", arg=scaled_graph)
    data_agent.save(DataType.SYNTHETIC_NETWORK_NKI, f"{prefix}_", arg=scaled_graph)

    scaled_fig = plot_scaled_network(scaled_graph)
    data_agent.saver.save_file(
        scaled_fig,
        DataType.SYNTHETIC_GRAPH,
        f"{prefix}_",
    )

    logger.info(
        f"Scaling complete — "
        f"{scaled_graph.number_of_nodes():,} nodes, "
        f"{scaled_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Efficient plotting for large networks
# ------------------------------------------------------------------ #
def plot_scaled_network(
    graph: SynthGraph,
    node_size: float = 0.05,
    line_width: float = 0.1,
    margin_frac: float = 0.02,
    dpi: int = 200,
) -> np.ndarray:
    """Plot the full scaled network using batched matplotlib primitives
    (``LineCollection`` + ``scatter``) so large graphs render efficiently.

    Returns an RGBA image array.
    """
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
    aspect = frame_w / frame_h if frame_h else 1.0
    fig_h = 12
    fig_w = fig_h * aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    segments = []
    for u, v in graph.edges():
        pu, pv = pos_arr[u], pos_arr[v]
        segments.append([pu, pv])
    lc = LineCollection(
        segments,
        colors="red",
        linewidths=line_width,
        zorder=2,
    )
    ax.add_collection(lc)

    ax.scatter(
        pos_arr[:, 0],
        pos_arr[:, 1],
        s=node_size,
        c="blue",
        zorder=3,
        edgecolors="none",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    return finalize_plot(fig, show=False)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_scaling_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
