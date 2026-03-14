# src/pipelines/hybrid.py
"""
Hybrid pipeline — parallel seed tiles (Phase 1) + frontier continuation (Phase 2).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Dict, List, Tuple

import networkit as nk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from analysis import MultifractalAnalyzer
from configs import BaseConfig
from graphs import GraphGenerator
from graphs._graph_node import GraphNode
from graphs.graph_generator import FrontierDescriptor
from graphs.synth_graph import SynthGraph
from handlers import AttributesCalculator, Mapper, RunAgent, Saver
from utils import build_graph, finalize_plot, save_hybrid_snapshot, trim_graph

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating…"


# ------------------------------------------------------------------ #
# Center placement — Poisson-disk-like rejection sampling
# ------------------------------------------------------------------ #


def generate_random_centers(
    whiteboard_w: float,
    whiteboard_h: float,
    min_distance: float,
    max_centers: int = 0,
    rng_seed: int = 42,
    max_rejections: int = 50_000,
) -> List[Tuple[float, float]]:
    """Place centers randomly with a minimum pairwise distance.

    Uses simple rejection sampling: draw a uniform point, accept if it
    is at least *min_distance* from every existing centre, otherwise
    discard.  Stops after *max_rejections* consecutive failures or
    when *max_centers* is reached (0 = no limit).
    """
    gen = np.random.RandomState(rng_seed)
    centers: List[Tuple[float, float]] = []
    consecutive_rejects = 0

    while consecutive_rejects < max_rejections:
        if max_centers > 0 and len(centers) >= max_centers:
            break
        x = gen.uniform(0, whiteboard_w)
        y = gen.uniform(0, whiteboard_h)
        too_close = False
        for cx, cy in centers:
            if (x - cx) ** 2 + (y - cy) ** 2 < min_distance**2:
                too_close = True
                break
        if too_close:
            consecutive_rejects += 1
            continue
        centers.append((x, y))
        consecutive_rejects = 0

    logger.info(
        f"Placed {len(centers):,} random centers "
        f"(whiteboard {whiteboard_w:.0f}×{whiteboard_h:.0f}, "
        f"min_dist={min_distance:.0f})"
    )
    return centers


# ------------------------------------------------------------------ #
# Phase 1 — parallel tile generation with quality control
# ------------------------------------------------------------------ #


def _generate_tile_worker(args):
    """Worker process: generate one seed tile with quality + frontier.

    Returns
    -------
    tile_idx, dict | None
        On success the dict contains:
        - ``error``    : float
        - ``positions``: list of (x, y)          — local coordinates
        - ``edges``    : list of ((x1,y1),…)     — local coordinates
        - ``frontier`` : list of FrontierDescriptor — local coordinates
    """
    import random

    tile_idx, exit_event, attributes, std_err_fea, mapper = args

    nk.setNumberOfThreads(1)

    seed = os.getpid() ^ tile_idx
    random.seed(seed)
    np.random.seed(seed % (2**31))

    if exit_event.is_set():
        return tile_idx, None

    try:
        GraphNode.initialize(attributes)

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
                return tile_idx, tile_payload

            if error < best_error:
                best_error = error
                best_result = tile_payload

        if best_result is not None:
            logger.warning(
                f"Tile {tile_idx}: max attempts reached "
                f"(best error={best_error:.4f})"
            )
            return tile_idx, best_result

        logger.error(f"Tile {tile_idx}: all attempts produced <100 nodes")
        return tile_idx, None

    except Exception as exc:
        logger.error(f"Tile {tile_idx} failed: {exc}", exc_info=True)
        return tile_idx, None


def run_phase1(
    attributes: AttributesCalculator,
    mapper: Mapper,
    std_err_fea,
    num_centers: int,
) -> Dict[int, dict]:
    """Generate all seed tiles in parallel and return their raw data."""
    nk.setNumberOfThreads(1)

    from multiprocessing import Manager

    exit_event = Manager().Event()
    num_workers = BaseConfig.get_max_workers(num_centers)

    logger.info(
        f"Phase 1: generating {num_centers:,} seed tiles " f"with {num_workers} workers"
    )

    tile_args = [
        (i, exit_event, attributes, std_err_fea, mapper) for i in range(num_centers)
    ]

    tile_results: Dict[int, dict] = {}
    failed = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_generate_tile_worker, a): a[0] for a in tile_args
            }
            completed = 0
            next_log_pct = 10
            for future in as_completed(futures):
                tile_idx, data = future.result()
                completed += 1
                progress = round(completed / num_centers * 100, 1)
                if progress >= next_log_pct:
                    err_str = f", error={data['error']:.4f}" if data else ""
                    logger.info(
                        f"Phase 1 progress: {progress}% "
                        f"({completed:,}/{num_centers:,}){err_str}"
                    )
                    next_log_pct += 10
                if data is not None:
                    tile_results[tile_idx] = data
                else:
                    failed.append(tile_idx)
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        exit_event.set()
        raise

    if failed:
        logger.warning(f"{len(failed)} tiles failed: {failed[:20]}")

    errors = [d["error"] for d in tile_results.values()]
    successful = sum(1 for e in errors if e < BaseConfig.ERROR_TOLERANCE)
    over_tol = len(tile_results) - successful
    logger.info(
        f"Phase 1 complete: {successful:,}/{num_centers:,} centers successful "
        f"(tol={BaseConfig.ERROR_TOLERANCE}), "
        f"{over_tol:,} over tolerance, {len(failed):,} failed, "
        f"avg error={np.mean(errors):.4f}"
    )
    return tile_results


# ------------------------------------------------------------------ #
# Phase 2 — assemble & continue
# ------------------------------------------------------------------ #


def run_phase2(
    tile_results: Dict[int, dict],
    attributes: AttributesCalculator,
    centers: List[Tuple[float, float]],
    whiteboard_w: float,
    whiteboard_h: float,
    max_rounds: int,
    snapshot_dir: str | None = None,
    snapshot_round_interval: int = 0,
) -> SynthGraph:
    """Offset tiles to their global center positions, then continue BFS.

    When *snapshot_round_interval* > 0, Phase 2 snapshots are rendered
    every N rounds.  When < 0, ~|N| log-spaced snapshots are taken.
    Snapshots are saved asynchronously in background threads to *snapshot_dir*.
    """
    frame_w, frame_h = BaseConfig.SYNTHETIC_FRAME_SIZE
    margin_x = frame_w
    margin_y = frame_h
    global_frame = (
        (-margin_x, whiteboard_w + margin_x),
        (-margin_y, whiteboard_h + margin_y),
    )

    take_snapshots = snapshot_round_interval != 0 and snapshot_dir is not None

    logger.info(
        f"Phase 2: whiteboard {whiteboard_w:.0f}×{whiteboard_h:.0f}, "
        f"global_frame={global_frame}"
        + (
            (
                f", snapshot every {snapshot_round_interval} round(s)"
                if snapshot_round_interval > 0
                else f", ~{abs(snapshot_round_interval)} log-spaced snapshots"
            )
            if take_snapshots
            else ""
        )
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

    GraphNode.initialize(attributes)

    if take_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_style = getattr(BaseConfig, "HYBRID_SNAPSHOT_STYLE", {})
        gen_workers = BaseConfig.get_max_workers()
        nproc = os.cpu_count() or 1
        plot_workers = max(nproc - gen_workers, gen_workers)
        plot_executor = ThreadPoolExecutor(max_workers=plot_workers)
        logger.info(f"Snapshot plot pool: {plot_workers} threads (nproc={nproc})")
        pending: List[Future] = []

        def on_snapshot(positions, edges, frame, idx):
            pos_arr = np.array(positions, dtype=np.float64)
            edge_arr = np.array(edges, dtype=np.float64)
            np.save(
                os.path.join(snapshot_dir, f"snapshot_{idx:05d}_positions.npy"), pos_arr
            )
            np.save(
                os.path.join(snapshot_dir, f"snapshot_{idx:05d}_edges.npy"), edge_arr
            )

            future = plot_executor.submit(
                save_hybrid_snapshot,
                positions,
                edges,
                frame,
                idx,
                snapshot_dir,
                **snapshot_style,
            )
            pending.append(future)
            logger.info(
                f"Snapshot {idx} queued: {len(positions):,} nodes, "
                f"{len(edges):,} edges  "
                f"(pending plots: {sum(1 for f in pending if not f.done())})"
            )

        graph = GraphGenerator.assemble_and_continue(
            tile_data_list,
            global_frame,
            max_rounds,
            snapshot_callback=on_snapshot,
            snapshot_round_interval=snapshot_round_interval,
        )

        remaining = sum(1 for f in pending if not f.done())
        if remaining:
            logger.info(f"Generation done. Waiting for {remaining} snapshot plot(s)...")
        for f in pending:
            f.result()
        plot_executor.shutdown(wait=False)
        logger.info(f"All {len(pending)} snapshot(s) saved to {snapshot_dir}")
    else:
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
    node_size: float = 0.01,
    line_width: float = 0.1,
    margin_frac: float = 0.02,
    dpi: int = None,
):
    from utils import recommend_dpi

    if dpi is None:
        dpi = recommend_dpi(graph.number_of_nodes())

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


def _write_report(path: str, title: str, num_nodes: int, num_edges: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"Nodes: {num_nodes:,}\n")
        f.write(f"Edges: {num_edges:,}\n")
    logger.info(f"Report saved: {path}")


def _apply_dataset_factors(dataset_id):
    """Apply per-dataset (nf, ef) overrides if configured."""
    overrides = getattr(BaseConfig, "DATASET_FACTORS", {})
    key = dataset_id[0]
    if key in overrides:
        nf, ef = overrides[key]
        BaseConfig.set_node_factor(nf)
        BaseConfig.set_edge_factor(ef)


def run_hybrid_for_dataset(dataset_id):
    logger.info(f"=== Hybrid pipeline for dataset: {dataset_id} ===")
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

    # --- Generate random centers ---
    centers = generate_random_centers(
        whiteboard_w, whiteboard_h, min_distance, max_centers=max_centers
    )

    # --- Phase 1 ---
    t0 = time.time()
    tile_results = run_phase1(attributes, mapper, std_err_fea, len(centers))
    logger.info(f"Phase 1 elapsed: {time.time() - t0:.1f}s")

    if not tile_results:
        logger.error("No tiles generated. Aborting.")
        return

    # --- Phase 2 (single-process: restore full networkit threading) ---
    max_threads = os.cpu_count() or 1
    nk.setNumberOfThreads(max_threads)
    logger.info(f"Phase 2: restored networkit threads to {max_threads}")

    snapshot_interval = getattr(BaseConfig, "SNAPSHOT_INTERVAL", 0)
    snapshot_dir = (
        os.path.join(data_agent.saver.output_dir, "snapshots")
        if snapshot_interval != 0
        else None
    )

    t1 = time.time()
    hybrid_graph = run_phase2(
        tile_results,
        attributes,
        centers,
        whiteboard_w,
        whiteboard_h,
        max_rounds,
        snapshot_dir=snapshot_dir,
        snapshot_round_interval=snapshot_interval,
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

    # --- Save original/ ---
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_graph")

    # --- Save synthetic/ ---
    data_agent.add_synthetic_graph(hybrid_graph)
    prefix = f"hybrid_{len(centers)}centers"

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
        f"Hybrid complete — "
        f"{hybrid_graph.number_of_nodes():,} nodes, "
        f"{hybrid_graph.number_of_edges():,} edges"
        + (f", snapshots in {snapshot_dir}" if snapshot_dir else "")
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main(config_cls=None):
    if config_cls is None:
        from configs.hybrid_mode import HybridConfig

        config_cls = HybridConfig
    config_cls.initialize()

    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_hybrid_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
