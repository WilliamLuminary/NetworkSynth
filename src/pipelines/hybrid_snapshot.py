# src/pipelines/hybrid_snapshot.py
"""
Hybrid snapshot pipeline — reuses Phase 1 from the standard hybrid
pipeline, then runs Phase 2 with round-by-round snapshot capture.

Does NOT modify or depend on any logic in ``hybrid.py`` beyond
reusing its Phase 1 runner and helper functions.
"""

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Tuple

import networkit as nk

from configs import BaseConfig
from configs.enums import DatasetId
from graphs import GraphGenerator
from graphs._graph_node import GraphNode
from graphs.graph_generator import FrontierDescriptor
from handlers import RunAgent, Saver
from pipelines.hybrid import (
    _apply_dataset_factors,
    _write_report,
    generate_random_centers,
    log_connectivity,
    plot_hybrid_network,
    run_phase1,
)
from utils import save_hybrid_snapshot, trim_graph

logger = logging.getLogger(__name__)


def run_phase2_with_snapshots(
    tile_results: Dict[int, dict],
    attributes,
    centers: List[Tuple[float, float]],
    whiteboard_w: float,
    whiteboard_h: float,
    max_rounds: int,
    snapshot_dir: str,
    snapshot_round_interval: int = 1,
):
    """Phase 2 with snapshot capture — mirrors hybrid.run_phase2."""
    frame_w, frame_h = BaseConfig.SYNTHETIC_FRAME_SIZE
    margin_x = frame_w
    margin_y = frame_h
    global_frame = (
        (-margin_x, whiteboard_w + margin_x),
        (-margin_y, whiteboard_h + margin_y),
    )

    logger.info(
        f"Phase 2 (snapshot): whiteboard {whiteboard_w:.0f}×{whiteboard_h:.0f}, "
        f"global_frame={global_frame}, "
        f"snapshot every {snapshot_round_interval} round(s)"
    )

    tile_data_list: List[dict] = []
    for tile_idx, data in tile_results.items():
        cx, cy = centers[tile_idx]

        offset_positions = [(x + cx, y + cy) for x, y in data["positions"]]
        offset_edges = [
            ((x1 + cx, y1 + cy), (x2 + cx, y2 + cy))
            for (x1, y1), (x2, y2) in data["edges"]
        ]
        offset_frontier = [
            FrontierDescriptor(
                position=(d.position[0] + cx, d.position[1] + cy),
                degree=d.degree,
                base_angle=d.base_angle,
                clockwise=d.clockwise,
                parent_position=(
                    (d.parent_position[0] + cx, d.parent_position[1] + cy)
                    if d.parent_position
                    else None
                ),
            )
            for d in data["frontier"]
        ]
        tile_data_list.append(
            {
                "positions": offset_positions,
                "edges": offset_edges,
                "frontier": offset_frontier,
            }
        )

    os.makedirs(snapshot_dir, exist_ok=True)

    plot_executor = ThreadPoolExecutor(max_workers=2)
    pending: List[Future] = []

    def on_snapshot(positions, edges, frame, idx):
        future = plot_executor.submit(
            save_hybrid_snapshot,
            positions,
            edges,
            frame,
            idx,
            snapshot_dir,
        )
        pending.append(future)
        logger.info(
            f"Snapshot {idx} queued: {len(positions):,} nodes, "
            f"{len(edges):,} edges  (pending plots: {sum(1 for f in pending if not f.done())})"
        )

    GraphNode.initialize(attributes)
    graph = GraphGenerator.assemble_and_continue(
        tile_data_list,
        global_frame,
        max_rounds,
        snapshot_callback=on_snapshot,
        snapshot_round_interval=snapshot_round_interval,
    )

    remaining = sum(1 for f in pending if not f.done())
    if remaining:
        logger.info(
            f"Generation done. Waiting for {remaining} snapshot plot(s) to finish..."
        )
    for f in pending:
        f.result()
    plot_executor.shutdown(wait=False)
    logger.info(f"All {len(pending)} snapshot(s) saved to {snapshot_dir}")

    return graph


def run_hybrid_snapshot_for_dataset(
    dataset_id: DatasetId,
    snapshot_round_interval: int = 1,
):
    """Run the hybrid pipeline for one dataset with Phase 2 snapshots."""
    from analysis import MultifractalAnalyzer

    logger.info(f"=== Hybrid snapshot pipeline for dataset: {dataset_id} ===")
    _apply_dataset_factors(dataset_id)
    logger.info(BaseConfig())

    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    attributes = data_agent.attributes
    mapper = data_agent.mapper

    std_err_fea = MultifractalAnalyzer(
        data_agent.get_original_network()
    ).analyze_error_features()

    scale_rows, scale_cols = BaseConfig.TARGET_SCALE
    max_rounds = BaseConfig.PHASE2_MAX_ROUNDS
    min_dist_factor = getattr(BaseConfig, "MIN_CENTER_DISTANCE_FACTOR", 1.5)

    frame_w, frame_h = BaseConfig.SYNTHETIC_FRAME_SIZE
    whiteboard_w = scale_cols * frame_w
    whiteboard_h = scale_rows * frame_h
    min_distance = min_dist_factor * max(frame_w, frame_h)

    max_centers = getattr(BaseConfig, "NUM_CENTERS", 0)

    centers = generate_random_centers(
        whiteboard_w, whiteboard_h, min_distance, max_centers=max_centers
    )

    # --- Phase 1 (reused from hybrid.py) ---
    t0 = time.time()
    tile_results = run_phase1(attributes, mapper, std_err_fea, len(centers))
    logger.info(f"Phase 1 elapsed: {time.time() - t0:.1f}s")

    if not tile_results:
        logger.error("No tiles generated. Aborting.")
        return

    # --- Phase 2 with snapshots ---
    max_threads = os.cpu_count() or 1
    nk.setNumberOfThreads(max_threads)

    snapshot_dir = os.path.join(data_agent.saver.output_dir, "snapshots")

    t1 = time.time()
    hybrid_graph = run_phase2_with_snapshots(
        tile_results,
        attributes,
        centers,
        whiteboard_w,
        whiteboard_h,
        max_rounds,
        snapshot_dir,
        snapshot_round_interval,
    )
    logger.info(f"Phase 2 elapsed: {time.time() - t1:.1f}s")

    hybrid_graph = trim_graph(hybrid_graph, attributes.average_degree)
    log_connectivity(hybrid_graph, "Pre-LCC")

    hybrid_graph = hybrid_graph.largest_connected_component()
    logger.info(
        f"After LCC: {hybrid_graph.number_of_nodes():,} nodes, "
        f"{hybrid_graph.number_of_edges():,} edges"
    )

    mapper.assign_weights(hybrid_graph)

    # --- Save original/ outputs ---
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_graph")

    # --- Save synthetic/ outputs ---
    data_agent.add_synthetic_graph(hybrid_graph)
    prefix = f"hybrid_snapshot_{len(centers)}centers"

    Saver.begin_batch()
    data_agent.saver.save(hybrid_graph, "synthetic_export", f"{prefix}_")
    fig = plot_hybrid_network(hybrid_graph)
    data_agent.saver.save(fig, "synthetic_graph", f"{prefix}_")
    Saver.end_batch()

    # --- Write reports ---
    original_network = data_agent.get_original_network()
    _write_report(
        os.path.join(data_agent.saver.output_dir, "original", "report.txt"),
        "Original Network",
        original_network.number_of_nodes(),
        original_network.number_of_edges(),
    )
    _write_report(
        os.path.join(data_agent.saver.output_dir, "synthetic", "report.txt"),
        "Synthetic Network",
        hybrid_graph.number_of_nodes(),
        hybrid_graph.number_of_edges(),
    )

    logger.info(
        f"Hybrid snapshot complete — "
        f"{hybrid_graph.number_of_nodes():,} nodes, "
        f"{hybrid_graph.number_of_edges():,} edges, "
        f"snapshots in {snapshot_dir}"
    )
