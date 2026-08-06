"""Diagnostic: measure MF error at 1×1 (partial) and full 3×3 scale.

Generates a few networks with no MF gating, then retroactively checks
what their MF errors would have been — both for the early check (~916
nodes) and the full-size check.  Reports timing for each stage.

Usage:
    cd NetworkSynth
    PYTHONPATH=src .venv/bin/python3 scripts/helpers/diagnose_mf_error.py
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "src")

from analysis import MultifractalAnalyzer
from configs.generate_mode.config_snapshot_3x3 import Snapshot3x3Config
from graphs import GraphGenerator
from handlers import RunAgent
from utils import trim_graph

N_SAMPLES = 3


def main():
    Snapshot3x3Config.ERROR_TOLERANCE = 0
    Snapshot3x3Config.initialize()

    from configs import BaseConfig
    from handlers import Saver

    Saver.initialize()
    dataset_id = BaseConfig.get_datasets()[0]
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()

    original = data_agent.get_original_network()
    original_node_count = original.number_of_nodes()
    logger.info(f"Original network: {original_node_count} nodes")

    t0 = time.perf_counter()
    std_err_fea = MultifractalAnalyzer(original).analyze_error_features()
    t_orig = time.perf_counter() - t0
    logger.info(f"Original MF features: {std_err_fea}  ({t_orig:.2f}s)")

    BaseConfig.seed_rng(0)
    generator = GraphGenerator(data_agent.attributes)

    for i in range(N_SAMPLES):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {i+1}/{N_SAMPLES}")
        logger.info(f"{'='*60}")

        partial_snapshot = {}

        def on_snapshot(positions, edges, frame, step_idx):
            n = len(positions)
            if n >= original_node_count and "partial" not in partial_snapshot:
                partial_snapshot["partial"] = (list(positions), set(edges), frame)

        t0 = time.perf_counter()
        graph = generator.generate_network_with_snapshots(
            snapshot_callback=on_snapshot,
            snapshot_interval=50,
        )
        t_bfs = time.perf_counter() - t0
        logger.info(
            f"BFS complete: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges  ({t_bfs:.2f}s)"
        )

        # --- Full-size MF check ---
        t0 = time.perf_counter()
        full_graph = trim_graph(graph, data_agent.attributes.average_degree)
        data_agent.mapper.assign_weights(full_graph)
        t_prep_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        full_err_fea = MultifractalAnalyzer(full_graph).analyze_error_features()
        full_error = MultifractalAnalyzer.analyze_error(full_err_fea, std_err_fea)
        t_mf_full = time.perf_counter() - t0
        logger.info(
            f"Full 3×3:  {full_graph.number_of_nodes()} nodes, "
            f"MF error = {full_error:.4f}, "
            f"prep={t_prep_full:.2f}s, MF={t_mf_full:.2f}s"
        )

        # --- Partial (~1×1) MF check ---
        if "partial" in partial_snapshot:
            positions, edges, frame = partial_snapshot["partial"]
            import numpy as np

            from graphs.synth_graph import SynthGraph

            pos_arr = np.array(positions)
            pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}
            edge_indices = []
            for p1, p2 in edges:
                u = pos_to_idx.get(p1)
                v = pos_to_idx.get(p2)
                if u is not None and v is not None:
                    edge_indices.append([u, v])
            edge_arr = (
                np.array(edge_indices) if edge_indices else np.empty((0, 2), dtype=int)
            )
            temp = SynthGraph.from_edge_list(pos_arr, edge_arr)
            temp = temp.largest_connected_component()

            t0 = time.perf_counter()
            temp = trim_graph(temp, data_agent.attributes.average_degree)
            data_agent.mapper.assign_weights(temp)
            t_prep_part = time.perf_counter() - t0

            t0 = time.perf_counter()
            part_err_fea = MultifractalAnalyzer(temp).analyze_error_features()
            part_error = MultifractalAnalyzer.analyze_error(part_err_fea, std_err_fea)
            t_mf_part = time.perf_counter() - t0
            logger.info(
                f"Partial 1×1: {temp.number_of_nodes()} nodes, "
                f"MF error = {part_error:.4f}, "
                f"prep={t_prep_part:.2f}s, MF={t_mf_part:.2f}s"
            )
        else:
            logger.info("No partial snapshot captured!")

        would_pass = full_error < 0.15
        logger.info(f"Would pass ERROR_TOLERANCE=0.15? {'YES' if would_pass else 'NO'}")

    logger.info(f"\n{'='*60}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
