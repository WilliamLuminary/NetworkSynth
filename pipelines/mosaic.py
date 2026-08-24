import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import networkit as nk

from configs import SynthParams
from configs.mosaic_mode.config_sample import SampleConfig as MosaicConfig
from graphs import GraphGenerator
from graphs.mosaic_stitcher import MosaicStitcher
from graphs.synth_graph import SynthGraph
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    AttributesCalculator,
    GenerationRun,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from utils import apply_seed, render_network, spawn_context, tagged, trim_graph

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process…"


# ------------------------------------------------------------------ #
# Tile generation (runs in worker processes)
# ------------------------------------------------------------------ #
def generate_single_tile(args):
    """
    Generate one tile network.

    Executed inside a worker process.  Params arrive in *args*, built by the
    parent, so nothing here depends on inherited class state.

    Parameters
    ----------
    args : tuple
        (row, col, exit_event, attributes, tile_gen_frame, offset_x, offset_y,
         params)

    Returns
    -------
    ((row, col), SynthGraph | None)
    """
    (
        row,
        col,
        exit_event,
        attributes,
        tile_gen_frame,
        offset_x,
        offset_y,
        params,
    ) = args

    # networkit defaults to one thread per core, and a forked child inherits an
    # OpenMP runtime whose threads do not exist in it — the child then spins
    # instead of working.  Matches hybrid's tile worker.
    nk.setNumberOfThreads(1)

    if exit_event.is_set():
        return (row, col), None

    try:
        apply_seed(params.seed)
        generator = GraphGenerator(attributes, params)
        graph = generator.generate_network(frame_range=tile_gen_frame)
        graph = trim_graph(graph, attributes.average_degree)

        positions = graph.positions()
        positions[:, 0] += offset_x
        positions[:, 1] += offset_y

        return (row, col), graph

    except Exception as exc:
        logger.error(f"Tile ({row}, {col}) failed: {exc}", exc_info=True)
        return (row, col), None


# ------------------------------------------------------------------ #
# Tile layout
# ------------------------------------------------------------------ #
def compute_tile_layout(config):
    """
    Compute the generation frame and global offset for every tile.

    The overlap margin on each side of a tile boundary equals
    ``OVERLAP_MARGIN_FRACTION × tile_dimension``, giving adjacent
    tiles a shared band in which nodes can be merged.

    Returns
    -------
    tile_gen_frame : (int, int)
        Width and height of the frame each tile is generated with
        (base tile size + overlap on each side).
    tile_offsets : dict[(row, col), (offset_x, offset_y)]
        Centre offset for every tile in global coordinates.
    """
    grid_rows = config.GRID_ROWS
    grid_cols = config.GRID_COLS
    tile_w, tile_h = config.TILE_FRAME_SIZE

    total_w = grid_cols * tile_w
    total_h = grid_rows * tile_h

    overlap_x = config.OVERLAP_MARGIN_FRACTION * tile_w
    overlap_y = config.OVERLAP_MARGIN_FRACTION * tile_h

    tile_gen_frame = (
        round(tile_w + 2 * overlap_x),
        round(tile_h + 2 * overlap_y),
    )

    logger.info(
        f"Mosaic layout: {grid_rows}×{grid_cols} grid, "
        f"tile=({tile_w}, {tile_h}), "
        f"overlap=({overlap_x:.1f}, {overlap_y:.1f}), "
        f"gen_frame={tile_gen_frame}, "
        f"total_frame=({total_w}, {total_h})"
    )

    tile_offsets: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for r in range(grid_rows):
        for c in range(grid_cols):
            offset_x = c * tile_w + tile_w / 2.0
            offset_y = r * tile_h + tile_h / 2.0
            tile_offsets[(r, c)] = (offset_x, offset_y)

    return tile_gen_frame, tile_offsets


# ------------------------------------------------------------------ #
# Parallel tile generation
# ------------------------------------------------------------------ #
def generate_all_tiles(
    attributes: AttributesCalculator,
    tile_gen_frame: Tuple[int, int],
    tile_offsets: Dict[Tuple[int, int], Tuple[float, float]],
    config,
) -> Dict[Tuple[int, int], SynthGraph]:
    """
    Generate every tile in parallel via ``ProcessPoolExecutor``.

    Returns
    -------
    dict[(row, col), SynthGraph]
        Successfully generated tiles with global-coordinate positions.
    """

    exit_event = spawn_context().Manager().Event()
    num_tiles = len(tile_offsets)
    num_workers = config.get_max_workers(num_tiles)

    logger.info(f"Generating {num_tiles:,} tiles with {num_workers} workers …")

    base_params = SynthParams.from_config(config)
    tile_args = [
        (
            r,
            c,
            exit_event,
            attributes,
            tile_gen_frame,
            off_x,
            off_y,
            base_params.for_worker(i),
        )
        for i, ((r, c), (off_x, off_y)) in enumerate(tile_offsets.items())
    ]

    tile_graphs: Dict[Tuple[int, int], SynthGraph] = {}
    failed_tiles = []

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=spawn_context()
        ) as executor:
            futures = {
                executor.submit(generate_single_tile, args): args[:2]
                for args in tile_args
            }

            completed = 0
            next_log_pct = 10
            for future in as_completed(futures):
                (row, col), graph = future.result()
                completed += 1

                progress = round(completed / num_tiles * 100, 1)
                if progress >= next_log_pct:
                    logger.info(
                        f"Progress: {progress}% ({completed:,}/{num_tiles:,})",
                        extra=tagged("PROGRESS", percent=progress),
                    )
                    next_log_pct += 10

                if graph is not None:
                    tile_graphs[(row, col)] = graph
                else:
                    failed_tiles.append((row, col))

    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        exit_event.set()
        raise

    if failed_tiles:
        logger.warning(
            f"{len(failed_tiles)} tiles failed: "
            f"{failed_tiles[:20]}{'…' if len(failed_tiles) > 20 else ''}"
        )

    logger.info(f"Generated {len(tile_graphs):,}/{num_tiles:,} tiles successfully")
    return tile_graphs


# ------------------------------------------------------------------ #
# Main pipeline for one dataset
# ------------------------------------------------------------------ #
def run_mosaic_for_dataset(dataset_id, config, run_paths):
    logger.info(f"=== Mosaic pipeline for dataset: {dataset_id} ===")
    logger.info(config())

    # 1. Load original network → compute structural attributes
    run = GenerationRun(config, run_paths, dataset_id)
    attributes = run.attributes

    # 2. Tile layout (positions + overlap)
    tile_gen_frame, tile_offsets = compute_tile_layout(config)

    # 3. Generate tiles in parallel
    tile_graphs = generate_all_tiles(attributes, tile_gen_frame, tile_offsets, config)
    if not tile_graphs:
        logger.error("No tiles were generated. Aborting.")
        return

    # 4. Stitch — merge close nodes across tile boundaries
    merge_threshold = attributes.average_length * config.CLOSED_NODES_FACTOR
    logger.info(f"Stitching with merge_threshold = {merge_threshold:.2f}")

    stitcher = MosaicStitcher(merge_threshold=merge_threshold)
    mosaic_graph = stitcher.stitch(tile_graphs)

    # 5. Assign edge weights from original network's length→weight distribution
    run.mapper.assign_weights(mosaic_graph)

    # 6. Save network data + plot
    run.add_synthetic_graph(mosaic_graph)
    prefix = f"mosaic_{config.GRID_ROWS}x{config.GRID_COLS}"
    run.saver.begin_batch()
    run.save(mosaic_graph, "synthetic_export", f"{prefix}_")
    # border=True outlines each tile's bounding box, which is the point of a
    # mosaic render and the only way it differs from every other one.
    mosaic_img = render_network(mosaic_graph, config.render("mosaic_graph"))
    run.save(mosaic_img, "synthetic_graph", f"{prefix}_")
    run.saver.end_batch()

    logger.info(
        f"Mosaic complete — "
        f"{mosaic_graph.number_of_nodes():,} nodes, "
        f"{mosaic_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Mosaic-aware plotting
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main(config_cls=MosaicConfig):
    config_cls.initialize()
    # Built before the try so the finally below can always name the run, even if
    # the very first dataset fails.
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            run_mosaic_for_dataset(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
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
