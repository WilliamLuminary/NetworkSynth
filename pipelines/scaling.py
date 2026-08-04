# src/pipelines/scaling.py
"""
Scaling pipeline — multi-root synchronized BFS for large networks.
"""

import logging

from configs import BaseConfig, SynthParams
from configs.scaling_mode import ScalingConfig
from graphs import GraphGenerator
from graphs.synth_graph import SynthGraph
from handlers import RunAgent, Saver
from utils import render_network, trim_graph

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
    generator = GraphGenerator(attributes, SynthParams.from_config(BaseConfig))
    scaled_graph = generator.generate_scaled_network(
        scale_rows=BaseConfig.SCALE_ROWS,
        scale_cols=BaseConfig.SCALE_COLS,
        max_rounds=BaseConfig.MAX_GENERATION_ROUNDS,
        root_spacing_factor=BaseConfig.ROOT_SPACING_FACTOR,
    )

    # 3. Trim to match average degree
    scaled_graph = trim_graph(scaled_graph, attributes.average_degree)

    # 4. Assign edge weights from original network's length→weight distribution
    data_agent.mapper.assign_weights(scaled_graph)

    # 5. Save network data + plot
    data_agent.add_synthetic_graph(scaled_graph)
    prefix = f"scaled_{BaseConfig.SCALE_ROWS}x{BaseConfig.SCALE_COLS}"

    Saver.begin_batch()
    data_agent.saver.save(scaled_graph, "synthetic_export", f"{prefix}_")
    scaled_img = plot_scaled_network(scaled_graph)
    data_agent.saver.save(scaled_img, "synthetic_graph", f"{prefix}_")
    Saver.end_batch()

    logger.info(
        f"Scaling complete — "
        f"{scaled_graph.number_of_nodes():,} nodes, "
        f"{scaled_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Efficient plotting for large networks
# ------------------------------------------------------------------ #
def plot_scaled_network(graph: SynthGraph, margin_frac: float = 0.02, dpi: int = None):
    """Render a scaled network to a PIL Image (CV2-backed, memory-safe)."""
    return render_network(graph, margin_frac=margin_frac, dpi=dpi)


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
