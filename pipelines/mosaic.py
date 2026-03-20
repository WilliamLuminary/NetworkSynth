# src/pipelines/mosaic.py
"""
Mosaic pipeline — generates a grid of small tile networks and stitches
them into one large network.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

from configs import BaseConfig
from configs.mosaic_mode import MosaicConfig
from graphs import GraphGenerator
from graphs.mosaic_stitcher import MosaicStitcher
from graphs.synth_graph import SynthGraph
from handlers import AttributesCalculator, RunAgent, Saver
from utils import render_network, trim_graph

MosaicConfig.initialize()

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process…"


# ------------------------------------------------------------------ #
# Tile generation (runs in worker processes)
# ------------------------------------------------------------------ #
def generate_single_tile(args):
    """
    Generate one tile network.

    Executed inside a worker process.  ``MosaicConfig.initialize()`` runs
    at module-import time, so every worker has ``BaseConfig`` set up
    correctly (CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR, etc.).

    Parameters
    ----------
    args : tuple
        (row, col, exit_event, attributes, tile_gen_frame, offset_x, offset_y)

    Returns
    -------
    ((row, col), SynthGraph | None)
    """
    row, col, exit_event, attributes, tile_gen_frame, offset_x, offset_y = args

    if exit_event.is_set():
        return (row, col), None

    try:
        generator = GraphGenerator(attributes)
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
def compute_tile_layout():
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
    grid_rows = BaseConfig.GRID_ROWS
    grid_cols = BaseConfig.GRID_COLS
    tile_w, tile_h = BaseConfig.TILE_FRAME_SIZE

    total_w = grid_cols * tile_w
    total_h = grid_rows * tile_h

    overlap_x = BaseConfig.OVERLAP_MARGIN_FRACTION * tile_w
    overlap_y = BaseConfig.OVERLAP_MARGIN_FRACTION * tile_h

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
) -> Dict[Tuple[int, int], SynthGraph]:
    """
    Generate every tile in parallel via ``ProcessPoolExecutor``.

    Returns
    -------
    dict[(row, col), SynthGraph]
        Successfully generated tiles with global-coordinate positions.
    """
    from multiprocessing import Manager

    exit_event = Manager().Event()
    num_tiles = len(tile_offsets)
    num_workers = BaseConfig.get_max_workers(num_tiles)

    logger.info(f"Generating {num_tiles:,} tiles with {num_workers} workers …")

    tile_args = [
        (r, c, exit_event, attributes, tile_gen_frame, off_x, off_y)
        for (r, c), (off_x, off_y) in tile_offsets.items()
    ]

    tile_graphs: Dict[Tuple[int, int], SynthGraph] = {}
    failed_tiles = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
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
                    logger.info(f"Progress: {progress}% ({completed:,}/{num_tiles:,})")
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
def run_mosaic_for_dataset(dataset_id):
    """Run the full mosaic pipeline for a single dataset."""
    logger.info(f"=== Mosaic pipeline for dataset: {dataset_id} ===")
    logger.info(BaseConfig())

    # 1. Load original network → compute structural attributes
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    attributes = data_agent.attributes

    # 2. Tile layout (positions + overlap)
    tile_gen_frame, tile_offsets = compute_tile_layout()

    # 3. Generate tiles in parallel
    tile_graphs = generate_all_tiles(attributes, tile_gen_frame, tile_offsets)
    if not tile_graphs:
        logger.error("No tiles were generated. Aborting.")
        return

    # 4. Stitch — merge close nodes across tile boundaries
    merge_threshold = attributes.average_length * BaseConfig.CLOSED_NODES_FACTOR
    logger.info(f"Stitching with merge_threshold = {merge_threshold:.2f}")

    stitcher = MosaicStitcher(merge_threshold=merge_threshold)
    mosaic_graph = stitcher.stitch(tile_graphs)

    # 5. Assign edge weights from original network's length→weight distribution
    data_agent.mapper.assign_weights(mosaic_graph)

    # 6. Save network data + plot
    data_agent.add_synthetic_graph(mosaic_graph)
    prefix = f"mosaic_{BaseConfig.GRID_ROWS}x{BaseConfig.GRID_COLS}"
    Saver.begin_batch()
    data_agent.saver.save(mosaic_graph, "synthetic_export", f"{prefix}_")
    mosaic_img = plot_mosaic_network(mosaic_graph)
    data_agent.saver.save(mosaic_img, "synthetic_graph", f"{prefix}_")
    Saver.end_batch()

    logger.info(
        f"Mosaic complete — "
        f"{mosaic_graph.number_of_nodes():,} nodes, "
        f"{mosaic_graph.number_of_edges():,} edges"
    )


# ------------------------------------------------------------------ #
# Mosaic-aware plotting
# ------------------------------------------------------------------ #
def plot_mosaic_network(graph: SynthGraph, margin_frac: float = 0.02, dpi: int = None):
    """Render a mosaic network to a PIL Image (CV2-backed, memory-safe).

    Includes a black border rectangle around the bounding box.
    """
    return render_network(graph, margin_frac=margin_frac, dpi=dpi, border=True)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_mosaic_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
