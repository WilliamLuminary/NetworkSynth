import gc
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
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import networkit as nk
import numpy as np

from analysis.error_checker import ErrorChecker, NullErrorChecker, create_error_checker
from configs import SynthParams
from graphs import GraphGenerator
from graphs._graph_node import GraphNode
from graphs.graph_generator import FrontierDescriptor
from graphs.synth_graph import SynthGraph
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    AttributesCalculator,
    GenerationRun,
    Mapper,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from utils import (
    apply_seed,
    build_graph,
    log_memory,
    render_network,
    save_hybrid_snapshot,
    spawn_context,
    tagged,
    trim_graph,
)

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating…"


# ------------------------------------------------------------------ #
# Center placement — Poisson-disk-like rejection sampling
# ------------------------------------------------------------------ #


def center_min_distance(config) -> float:
    """How close two seed centers are allowed to land.

    Read both by the sampler that places them and by the frame sizing that
    measures the gaps afterwards, so the two cannot drift apart.
    """
    return config.MIN_CENTER_DISTANCE_FACTOR * max(config.FRAME_SIZE)


def generate_random_centers(
    whiteboard_w: float,
    whiteboard_h: float,
    min_distance: float,
    max_centers: int = 0,
    rng_seed: Optional[int] = None,
    max_rejections: int = 50_000,
) -> List[Tuple[float, float]]:
    """Place centers randomly with a minimum pairwise distance.

    Uses simple rejection sampling: draw a uniform point, accept if it
    is at least *min_distance* from every existing centre, otherwise
    discard.  Stops after *max_rejections* consecutive failures or
    when *max_centers* is reached (0 = no limit).

    *rng_seed* is the run's ``SEED``, so where the centers land follows the
    same switch as everything else a run draws: a seed replays the same
    scatter, ``None`` scatters differently every time.
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
        f"min_dist={min_distance:.0f})",
        extra=tagged("PHASE1"),
    )
    return centers


def compute_center_frames(
    centers: List[Tuple[float, float]],
    config,
) -> List[Tuple[float, float]]:
    """Decide how big each seed tile's growth frame should be.

    Two modes, selected by config:

    * **Fixed** — if ``TILE_FRAME_SIZE`` is set, every tile uses that one
      frame. Fully deterministic, no measurement.
    * **Auto** — otherwise, each tile is sized from the distance ``d`` to
      its nearest neighboring center: frame side = ``d * TILE_FRAME_FACTOR``
      (default 0.5), with a floor of the same rule applied to the closest
      two centers the sampler will place — ``center_min_distance *
      TILE_FRAME_FACTOR`` — so no tile ends up smaller than the spacing
      allows.  Measured rather than configured: a floor in pixels means
      nothing without knowing the frame it is a fraction of.

    Falls back to ``SYNTHETIC_FRAME_SIZE`` when fewer than two centers
    exist (no neighbor to measure).
    """
    fixed = config.TILE_FRAME_SIZE
    if fixed is not None:
        logger.info(f"Tile frames: fixed {fixed}", extra=tagged("PHASE1"))
        return [tuple(fixed)] * len(centers)

    if len(centers) < 2:
        return [config.SYNTHETIC_FRAME_SIZE] * len(centers)

    factor = config.TILE_FRAME_FACTOR
    floor = center_min_distance(config) * factor

    from scipy.spatial import cKDTree

    pts = np.asarray(centers, dtype=np.float64)
    dist, _ = cKDTree(pts).query(pts, k=2)
    nearest = dist[:, 1]
    sides = np.maximum(nearest * factor, floor)
    frames = [(s, s) for s in sides]

    logger.info(
        f"Tile frames: auto (factor={factor}, floor={floor:.0f}); "
        f"nearest-neighbor dist min={nearest.min():.0f}, "
        f"mean={nearest.mean():.0f}, max={nearest.max():.0f}; "
        f"frame side min={sides.min():.0f}, max={sides.max():.0f}",
        extra=tagged("PHASE1"),
    )
    return frames


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
    (
        tile_idx,
        exit_event,
        attributes,
        error_checker,
        mapper,
        frame_range,
        params,
        min_tile_nodes,
    ) = args

    nk.setNumberOfThreads(1)

    # Seeded from the params this worker was handed, so a seeded run repeats.
    # When unseeded this is a no-op: pool children diverge on their own.
    apply_seed(params.seed)

    if exit_event.is_set():
        return tile_idx, None

    try:
        with GraphNode.traversal(attributes, params):
            best_error = float("inf")
            best_result = None

            for attempt in range(params.max_attempts):
                if exit_event.is_set():
                    break

                result = GraphGenerator._bfs_network_with_frontier(frame_range)
                (
                    inner_nodes,
                    inner_edges,
                    frontier_descs,
                    all_positions,
                    all_edge_tuples,
                ) = result

                if not inner_nodes or len(inner_nodes) < min_tile_nodes:
                    continue

                # Skip quality analysis entirely when no checker is configured.
                if isinstance(error_checker, NullErrorChecker):
                    return tile_idx, {
                        "error": 0.0,
                        "positions": all_positions,
                        "edges": all_edge_tuples,
                        "frontier": frontier_descs,
                    }

                graph = build_graph(inner_nodes, inner_edges, arg_type="graph_node")
                graph = trim_graph(graph, attributes.average_degree)
                mapper.assign_weights(graph)

                passed, error = error_checker.check(graph)

                tile_payload = {
                    "error": error,
                    "positions": all_positions,
                    "edges": all_edge_tuples,
                    "frontier": frontier_descs,
                }

                if passed:
                    return tile_idx, tile_payload

                if error < best_error:
                    best_error = error
                    best_result = tile_payload

            if best_result is not None:
                logger.warning(
                    f"Tile {tile_idx}: max attempts reached "
                    f"(best error={best_error:.4f})",
                    extra=tagged("TILE"),
                )
                return tile_idx, best_result

            logger.error(
                f"Tile {tile_idx}: all attempts produced <{min_tile_nodes} nodes",
                extra=tagged("TILE"),
            )
            return tile_idx, None

    except Exception as exc:
        logger.error(
            f"Tile {tile_idx} failed: {exc}", exc_info=True, extra=tagged("TILE")
        )
        return tile_idx, None


def run_phase1(
    attributes: AttributesCalculator,
    mapper: Mapper,
    error_checker: ErrorChecker,
    frames: List[Tuple[float, float]],
    config,
    base_params: SynthParams,
) -> Dict[int, dict]:
    """Generate all seed tiles in parallel and return their raw data.

    *frames* holds the per-center frame box side for every tile; its
    length is the number of centers to generate.
    """
    nk.setNumberOfThreads(1)

    num_centers = len(frames)
    manager = spawn_context().Manager()
    exit_event = manager.Event()
    num_workers = config.get_max_workers(num_centers)

    logger.info(
        f"Phase 1: generating {num_centers:,} seed tiles with {num_workers} workers",
        extra=tagged("PHASE1"),
    )

    # Params derived per worker, so each tile is independently seeded and
    # nothing relies on inherited class state.
    tile_args = [
        (
            i,
            exit_event,
            attributes,
            error_checker,
            mapper,
            frames[i],
            base_params.for_worker(i),
            config.MIN_TILE_NODES,
        )
        for i in range(num_centers)
    ]

    tile_results: Dict[int, dict] = {}
    failed = []

    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=spawn_context())
    interrupted = False
    try:
        futures = {executor.submit(_generate_tile_worker, a): a[0] for a in tile_args}
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
                    f"({completed:,}/{num_centers:,}){err_str}",
                    extra=tagged("PHASE1", percent=progress),
                )
                next_log_pct += 10
            if data is not None:
                tile_results[tile_idx] = data
            else:
                failed.append(tile_idx)
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        exit_event.set()
        interrupted = True
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
        manager.shutdown()

    if failed:
        logger.warning(
            f"{len(failed)} tiles failed: {failed[:20]}",
            extra=tagged("PHASE1"),
        )

    errors = [d["error"] for d in tile_results.values()]
    successful = sum(1 for e in errors if e < config.ERROR_TOLERANCE)
    over_tol = len(tile_results) - successful
    logger.info(
        f"Phase 1 complete: {successful:,}/{num_centers:,} centers successful "
        f"(tol={config.ERROR_TOLERANCE}), "
        f"{over_tol:,} over tolerance, {len(failed):,} failed, "
        f"avg error={np.mean(errors):.4f}",
        extra=tagged("PHASE1"),
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
    config,
    params: SynthParams,
    snapshot_dir: str | None = None,
    snapshot_round_interval: int = 0,
) -> SynthGraph:
    """Offset tiles to their global center positions, then continue BFS.

    When *snapshot_round_interval* > 0, Phase 2 snapshots are rendered
    every N rounds.  When < 0, ~|N| log-spaced snapshots are taken.
    Snapshots are saved asynchronously in background threads to *snapshot_dir*.
    """
    frame_w, frame_h = config.SYNTHETIC_FRAME_SIZE
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
        ),
        extra=tagged("PHASE2"),
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

    apply_seed(params.seed)

    if take_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)
        # Resolved here in the parent: the plot pool runs in child processes.
        snapshot_style = config.render("hybrid_snapshot")
        plot_workers = config.get_snapshot_plot_workers()
        plot_executor = ThreadPoolExecutor(max_workers=plot_workers)
        logger.info(
            f"Snapshot plot pool: {plot_workers} threads", extra=tagged("SNAPSHOT")
        )
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
                snapshot_style,
            )
            pending.append(future)
            logger.info(
                f"Snapshot {idx} queued: {len(positions):,} nodes, "
                f"{len(edges):,} edges  "
                f"(pending plots: {sum(1 for f in pending if not f.done())})",
                extra=tagged("SNAPSHOT"),
            )

        with GraphNode.traversal(attributes, params):
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
                f"Generation done. Waiting for {remaining} snapshot plot(s)...",
                extra=tagged("SNAPSHOT"),
            )
        for f in pending:
            f.result()
        num_snapshots = len(pending)
        plot_executor.shutdown(wait=True)
        del pending, plot_executor
        logger.info(
            f"All {num_snapshots} snapshot(s) saved to {snapshot_dir}",
            extra=tagged("SNAPSHOT"),
        )
    else:
        with GraphNode.traversal(attributes, params):
            graph = GraphGenerator.assemble_and_continue(
                tile_data_list, global_frame, max_rounds
            )

    return graph


# ------------------------------------------------------------------ #
# Connectivity diagnostics
# ------------------------------------------------------------------ #


def log_connectivity(graph: SynthGraph, label: str = ""):
    cc = nk.components.ConnectedComponents(graph.nk)
    cc.run()
    sizes = sorted(cc.getComponentSizes().values(), reverse=True)
    n = graph.number_of_nodes()
    logger.info(
        f"{label} connectivity: {cc.numberOfComponents()} components, "
        f"LCC={sizes[0]:,} ({sizes[0] / n * 100:.2f}% of {n:,} nodes)",
        extra=tagged("STATS"),
    )
    if len(sizes) > 1:
        logger.info(f"  top-5 sizes: {sizes[:5]}", extra=tagged("STATS"))


# ------------------------------------------------------------------ #
# Plotting (reused from scaling mode)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# Main pipeline for one dataset
# ------------------------------------------------------------------ #


def _apply_dataset_factors(dataset_id, config, params: SynthParams) -> SynthParams:
    """Return *params* with this dataset's (nf, ef) override applied, if any.

    Returns a new params object rather than mutating config, so per-dataset
    factors cannot leak into another dataset or another run.
    """
    overrides = config.DATASET_FACTORS
    key = dataset_id[0]
    if key not in overrides:
        return params
    nf, ef = overrides[key]
    logger.info(f"Dataset {key}: factors nf={nf}, ef={ef}", extra=tagged("CONFIG"))
    return replace(params, closed_nodes_factor=nf, closed_edges_factor=ef)


def run_hybrid_for_dataset(dataset_id, config, run_paths):
    logger.info(
        f"=== Hybrid pipeline for dataset: {dataset_id} ===",
        extra=tagged("PIPELINE", dataset=str(dataset_id)),
    )
    log_memory(f"Start dataset {dataset_id}", config.LOG_MEMORY)
    base_params = _apply_dataset_factors(
        dataset_id, config, SynthParams.from_config(config)
    )
    logger.info(config())

    run = GenerationRun(config, run_paths, dataset_id)
    attributes = run.attributes
    mapper = run.mapper

    error_checker = create_error_checker(config)
    error_checker.compute_reference(run.original)

    scale_rows, scale_cols = config.TARGET_SCALE
    max_rounds = config.PHASE2_MAX_ROUNDS

    # The frame, not IMAGE_SIZE.  A tile's coordinates are in frame space, and
    # every config that declared both set IMAGE_SIZE to the frame transposed —
    # config_dickson says so in a comment — so this reads the same numbers by
    # the name that means them.  IMAGE_SIZE is the input image's true size now,
    # and nothing lays out by it.
    frame_w, frame_h = config.FRAME_SIZE
    whiteboard_w = scale_cols * frame_w
    whiteboard_h = scale_rows * frame_h
    min_distance = center_min_distance(config)

    max_centers = config.NUM_CENTERS

    # --- Generate random centers ---
    centers = generate_random_centers(
        whiteboard_w,
        whiteboard_h,
        min_distance,
        max_centers=max_centers,
        rng_seed=config.SEED,
    )
    frames = compute_center_frames(centers, config)

    # --- Phase 1 ---
    log_memory(f"Before Phase 1 ({dataset_id})", config.LOG_MEMORY)
    t0 = time.time()
    tile_results = run_phase1(
        attributes, mapper, error_checker, frames, config, base_params
    )
    logger.info(
        f"Phase 1 elapsed: {time.time() - t0:.1f}s",
        extra=tagged("PHASE1", dataset=str(dataset_id)),
    )
    log_memory(f"After Phase 1 ({dataset_id})", config.LOG_MEMORY)

    if not tile_results:
        logger.error(
            "No tiles generated. Aborting.",
            extra=tagged("PHASE1", dataset=str(dataset_id)),
        )
        return

    # --- Phase 2 (single-process: restore full networkit threading) ---
    max_threads = os.cpu_count() or 1
    nk.setNumberOfThreads(max_threads)
    logger.info(
        f"Phase 2: restored networkit threads to {max_threads}",
        extra=tagged("PHASE2", dataset=str(dataset_id)),
    )

    snapshot_interval = config.SNAPSHOT_INTERVAL
    snapshot_dir = (
        os.path.join(run.saver.output_dir, "snapshots")
        if snapshot_interval != 0
        else None
    )

    log_memory(f"Before Phase 2 ({dataset_id})", config.LOG_MEMORY)
    t1 = time.time()
    hybrid_graph = run_phase2(
        tile_results,
        attributes,
        centers,
        whiteboard_w,
        whiteboard_h,
        max_rounds,
        config,
        base_params,
        snapshot_dir=snapshot_dir,
        snapshot_round_interval=snapshot_interval,
    )
    logger.info(
        f"Phase 2 elapsed: {time.time() - t1:.1f}s",
        extra=tagged("PHASE2", dataset=str(dataset_id)),
    )
    log_memory(f"After Phase 2 ({dataset_id})", config.LOG_MEMORY)

    # --- Free Phase 1 tile data before post-processing ---
    # Phase 2's traversal already dropped the node and edge grids on its way
    # out; this is only the tile payloads.
    del tile_results
    gc.collect()
    log_memory(f"After Phase 2 cleanup ({dataset_id})", config.LOG_MEMORY)

    hybrid_graph = trim_graph(hybrid_graph, attributes.average_degree)
    log_connectivity(hybrid_graph, "Pre-LCC")

    hybrid_graph = hybrid_graph.largest_connected_component()
    num_nodes = hybrid_graph.number_of_nodes()
    num_edges = hybrid_graph.number_of_edges()
    logger.info(
        f"After LCC: {num_nodes:,} nodes, {num_edges:,} edges",
        extra=tagged("STATS", dataset=str(dataset_id)),
    )

    mapper.assign_weights(hybrid_graph)

    # --- Save original/ ---
    run.save_original()

    # --- Save synthetic/ ---
    run.add_synthetic_graph(hybrid_graph)
    prefix = f"hybrid_{len(centers)}centers"

    run.saver.begin_batch()
    run.save(hybrid_graph, "synthetic_export", f"{prefix}_")

    # --- Write reports before plotting (plotting is memory-intensive) ---
    run.save(run.original_report(), "original_report")
    run.save(run.synthetic_report(hybrid_graph), "synthetic_report")

    # --- Plot synthetic graph (high memory) ---
    # Free everything we can before rendering.
    saver = run.saver
    del run, attributes, mapper
    del error_checker, centers
    gc.collect()

    log_memory(f"Before plotting ({dataset_id})", config.LOG_MEMORY)
    img = render_network(hybrid_graph, config.render("hybrid_graph"))
    del hybrid_graph
    gc.collect()
    saver.save(img, "synthetic_graph", f"{prefix}_")
    saver.end_batch()
    del img, saver
    gc.collect()

    log_memory(f"End dataset {dataset_id}", config.LOG_MEMORY)
    logger.info(
        f"Hybrid complete — "
        f"{num_nodes:,} nodes, {num_edges:,} edges"
        + (f", snapshots in {snapshot_dir}" if snapshot_dir else ""),
        extra=tagged("PIPELINE", dataset=str(dataset_id)),
    )
    logger.info(
        "Resources released for dataset %s",
        dataset_id,
        extra=tagged("PIPELINE", dataset=str(dataset_id)),
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def _subprocess_target(dataset_id, config, run_paths, run_id):
    """Top-level target for child processes (must be picklable for spawn).

    The child sets the config up from scratch.  Nothing the parent did to the
    class travels: a module-level config pickles as a name reference, and a
    run-spec config is rebuilt from its spec — either way the loader functions
    ``initialize()`` installs are absent, and calling the pipeline without them
    fails on ``load_idle``.  Under ``fork`` this was invisible, because the
    child inherited the initialised class instead of rebuilding it.

    *run_id* comes from the parent so the child's log lines land in the same
    ``run.jsonl``, under the same id, rather than in a run of their own.
    """
    config.RUN_ID = run_id
    config.initialize()
    attach_run_log(run_paths.root, run_id)
    try:
        run_hybrid_for_dataset(dataset_id, config, run_paths)
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)


def _run_dataset_in_subprocess(dataset_id, config, run_paths):
    """Run a single dataset in a child process for full memory isolation.

    When the child exits, the OS reclaims *all* of its memory — no
    fragmentation carries over to the next dataset.
    """
    proc = spawn_context().Process(
        target=_subprocess_target,
        args=(dataset_id, config, run_paths, config.RUN_ID),
        name=f"hybrid-{dataset_id}",
    )
    proc.start()

    try:
        proc.join()
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        # Give child a chance to shut down cleanly
        proc.join(timeout=5)
        if proc.is_alive():
            logger.warning("Child process did not exit in time, terminating…")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        raise

    if proc.exitcode != 0:
        logger.error(
            f"Dataset {dataset_id} subprocess exited with code {proc.exitcode}",
            extra=tagged("PIPELINE", dataset=str(dataset_id)),
        )
        # Raised, not only logged: main() decides the run's status from what
        # reaches it, so a dataset that died in its child would otherwise be
        # written up as a completed run that happens to have no output.
        raise RuntimeError(
            f"dataset {dataset_id} subprocess exited with code {proc.exitcode}"
        )
    else:
        logger.info(
            f"Dataset {dataset_id} subprocess finished successfully",
            extra=tagged("PIPELINE", dataset=str(dataset_id)),
        )
    log_memory(f"Main process after {dataset_id} subprocess", config.LOG_MEMORY)


def main(config_cls=None):
    if config_cls is None:
        from configs.hybrid_mode.config_sample import SampleConfig as HybridConfig

        config_cls = HybridConfig
    config_cls.initialize()

    # Built before the try so the finally below can always name the run, even if
    # the very first dataset fails.
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            _run_dataset_in_subprocess(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.info("Shutdown complete.")
        # Re-raised so the entry point can exit non-zero: a cancelled
        # run must not look like a completed one to a caller.
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Written even on cancel/failure: a caller must be able to tell
        # "manifest says cancelled" from "no manifest, we died hard".
        if not config_cls.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
