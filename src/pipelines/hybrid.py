# src/pipelines/hybrid.py
"""
Hybrid pipeline — parallel seed tiles (Phase 1) + frontier continuation (Phase 2).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import networkit as nk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from analysis import MultifractalAnalyzer
from configs import BaseConfig, DataType
from configs.hybrid_mode import HybridConfig
from graphs import GraphGenerator
from graphs.graph_generator import FrontierDescriptor
from graphs.synth_graph import SynthGraph
from handlers import AttributesCalculator, Mapper, RunAgent, Saver
from utils import build_graph, finalize_plot, trim_graph

HybridConfig.initialize()

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating…"


# ------------------------------------------------------------------ #
# Phase 1 — parallel tile generation with quality control
# ------------------------------------------------------------------ #


def _generate_tile_worker(args):
    """Worker process: generate one seed tile with quality + frontier.

    Returns
    -------
    (row, col), dict | None
        On success the dict contains:
        - ``error``    : float
        - ``positions``: list of (x, y)          — local coordinates
        - ``edges``    : list of ((x1,y1),…)     — local coordinates
        - ``frontier`` : list of FrontierDescriptor — local coordinates
    """
    import random

    row, col, exit_event, attributes, std_err_fea, mapper = args

    seed = os.getpid() ^ (row * 1000 + col)
    random.seed(seed)
    np.random.seed(seed % (2**31))

    if exit_event.is_set():
        return (row, col), None

    try:
        best_error = float("inf")
        best_result = None

        for attempt in range(BaseConfig.MAX_ATTEMPTS):
            result = GraphGenerator._bfs_network_with_frontier()
            inner_nodes, inner_edges, frontier_descs, all_positions, all_edge_tuples = (
                result
            )

            if not inner_nodes or len(inner_nodes) < 100:
                continue

            graph = build_graph(inner_nodes, inner_edges, arg_type="graph_node")
            graph = trim_graph(graph, attributes.average_degree)
            mapper.assign_weights(graph)

            err_fea = MultifractalAnalyzer(graph).analyze_error_features()
            error = MultifractalAnalyzer.analyze_error(err_fea, std_err_fea)

            tile_payload = {
                "error": error,
                "positions": all_positions,
                "edges": all_edge_tuples,
                "frontier": frontier_descs,
            }

            if error < BaseConfig.ERROR_TOLERANCE:
                return (row, col), tile_payload

            if error < best_error:
                best_error = error
                best_result = tile_payload

        if best_result is not None:
            logger.warning(
                f"Tile ({row},{col}): max attempts reached "
                f"(best error={best_error:.4f})"
            )
            return (row, col), best_result

        logger.error(f"Tile ({row},{col}): all attempts produced <100 nodes")
        return (row, col), None

    except Exception as exc:
        logger.error(f"Tile ({row},{col}) failed: {exc}", exc_info=True)
        return (row, col), None


def run_phase1(
    attributes: AttributesCalculator,
    mapper: Mapper,
    std_err_fea,
    rows: int,
    cols: int,
) -> Dict[Tuple[int, int], dict]:
    """Generate all seed tiles in parallel and return their raw data."""
    from multiprocessing import Manager

    exit_event = Manager().Event()
    num_tiles = rows * cols
    num_workers = BaseConfig.get_max_workers(num_tiles)

    logger.info(
        f"Phase 1: generating {num_tiles:,} seed tiles "
        f"({rows}×{cols}) with {num_workers} workers"
    )

    tile_args = [
        (r, c, exit_event, attributes, std_err_fea, mapper)
        for r in range(rows)
        for c in range(cols)
    ]

    tile_results: Dict[Tuple[int, int], dict] = {}
    failed = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_tile_worker, a): a[:2] for a in tile_args
            }
            completed = 0
            next_log_pct = 10
            for future in as_completed(futures):
                (r, c), data = future.result()
                completed += 1
                progress = round(completed / num_tiles * 100, 1)
                if progress >= next_log_pct:
                    err_str = f", error={data['error']:.4f}" if data else ""
                    logger.info(
                        f"Phase 1 progress: {progress}% "
                        f"({completed:,}/{num_tiles:,}){err_str}"
                    )
                    next_log_pct += 10
                if data is not None:
                    tile_results[(r, c)] = data
                else:
                    failed.append((r, c))
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        exit_event.set()
        raise

    if failed:
        logger.warning(f"{len(failed)} tiles failed: {failed[:20]}")

    errors = [d["error"] for d in tile_results.values()]
    logger.info(
        f"Phase 1 complete: {len(tile_results):,}/{num_tiles:,} tiles OK, "
        f"avg error={np.mean(errors):.4f}"
    )
    return tile_results


# ------------------------------------------------------------------ #
# Phase 2 — assemble & continue
# ------------------------------------------------------------------ #


def run_phase2(
    tile_results: Dict[Tuple[int, int], dict],
    attributes: AttributesCalculator,
    rows: int,
    cols: int,
    max_rounds: int,
) -> SynthGraph:
    """Offset tiles into global coordinates, then assemble and continue BFS."""
    frame_w, frame_h = BaseConfig.SYNTHETIC_FRAME_SIZE
    spacing_w = frame_w * 2.0
    spacing_h = frame_h * 2.0

    global_w = cols * spacing_w
    global_h = rows * spacing_h
    margin_x = spacing_w * 0.5
    margin_y = spacing_h * 0.5
    global_frame = (
        (-margin_x, global_w + margin_x),
        (-margin_y, global_h + margin_y),
    )

    logger.info(
        f"Phase 2: spacing=({spacing_w:.0f},{spacing_h:.0f}), "
        f"global_frame={global_frame}"
    )

    tile_data_list: List[dict] = []
    for (r, c), data in tile_results.items():
        offset_x = c * spacing_w + spacing_w / 2.0
        offset_y = r * spacing_h + spacing_h / 2.0

        offset_positions = [(x + offset_x, y + offset_y) for x, y in data["positions"]]
        offset_edges = [
            (
                (x1 + offset_x, y1 + offset_y),
                (x2 + offset_x, y2 + offset_y),
            )
            for (x1, y1), (x2, y2) in data["edges"]
        ]
        offset_frontier = [
            FrontierDescriptor(
                position=(d.position[0] + offset_x, d.position[1] + offset_y),
                degree=d.degree,
                base_angle=d.base_angle,
                clockwise=d.clockwise,
                parent_position=(
                    (d.parent_position[0] + offset_x, d.parent_position[1] + offset_y)
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

    graph = GraphGenerator.assemble_and_continue(
        tile_data_list, global_frame, max_rounds
    )
    return graph


# ------------------------------------------------------------------ #
# Connectivity diagnostics
# ------------------------------------------------------------------ #


def log_connectivity(graph: SynthGraph, label: str = ""):
    """Log connected-component statistics."""
    cc = nk.components.ConnectedComponents(graph.nk)
    cc.run()
    sizes = sorted(cc.getComponentSizes().values(), reverse=True)
    n = graph.number_of_nodes()
    logger.info(
        f"{label} connectivity: {cc.numberOfComponents()} components, "
        f"LCC={sizes[0]:,} ({sizes[0]/n*100:.2f}% of {n:,} nodes)"
    )
    if len(sizes) > 1:
        logger.info(f"  top-5 sizes: {sizes[:5]}")


# ------------------------------------------------------------------ #
# Plotting (reused from scaling mode)
# ------------------------------------------------------------------ #


def plot_hybrid_network(
    graph: SynthGraph,
    node_size: float = 0.05,
    line_width: float = 0.1,
    margin_frac: float = 0.02,
    dpi: int = 200,
) -> np.ndarray:
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
    lc = LineCollection(segments, colors="red", linewidths=line_width, zorder=2)
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
# Main pipeline for one dataset
# ------------------------------------------------------------------ #


def run_hybrid_for_dataset(dataset_id):
    logger.info(f"=== Hybrid pipeline for dataset: {dataset_id} ===")
    logger.info(BaseConfig())

    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    attributes = data_agent.attributes
    mapper = data_agent.mapper

    std_err_fea = MultifractalAnalyzer(
        data_agent.get_original_network()
    ).analyze_error_features()

    rows = BaseConfig.HYBRID_ROWS
    cols = BaseConfig.HYBRID_COLS
    max_rounds = BaseConfig.PHASE2_MAX_ROUNDS

    # --- Phase 1 ---
    t0 = time.time()
    tile_results = run_phase1(attributes, mapper, std_err_fea, rows, cols)
    logger.info(f"Phase 1 elapsed: {time.time() - t0:.1f}s")

    if not tile_results:
        logger.error("No tiles generated. Aborting.")
        return

    # --- Phase 2 (single-process: restore full networkit threading) ---
    max_threads = os.cpu_count() or 1
    nk.setNumberOfThreads(max_threads)
    logger.info(f"Phase 2: restored networkit threads to {max_threads}")

    t1 = time.time()
    hybrid_graph = run_phase2(tile_results, attributes, rows, cols, max_rounds)
    logger.info(f"Phase 2 elapsed: {time.time() - t1:.1f}s")

    hybrid_graph = trim_graph(hybrid_graph, attributes.average_degree)
    log_connectivity(hybrid_graph, "Pre-LCC")

    hybrid_graph = hybrid_graph.largest_connected_component()
    logger.info(
        f"After LCC: {hybrid_graph.number_of_nodes():,} nodes, "
        f"{hybrid_graph.number_of_edges():,} edges"
    )

    # --- Save ---
    data_agent.add_synthetic_graph(hybrid_graph)
    prefix = f"hybrid_{rows}x{cols}"

    data_agent.save(DataType.SYNTHETIC_EDGELIST, f"{prefix}_", arg=hybrid_graph)
    data_agent.save(DataType.SYNTHETIC_POSITIONS, f"{prefix}_", arg=hybrid_graph)
    data_agent.save(DataType.SYNTHETIC_NETWORK_NKI, f"{prefix}_", arg=hybrid_graph)

    fig = plot_hybrid_network(hybrid_graph)
    data_agent.saver.save_file(fig, DataType.SYNTHETIC_GRAPH, f"{prefix}_")

    logger.info(
        f"Hybrid complete — "
        f"{hybrid_graph.number_of_nodes():,} nodes, "
        f"{hybrid_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main():
    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_hybrid_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
