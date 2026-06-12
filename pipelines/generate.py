# src/pipelines/generate.py
"""
Core network generation pipeline.

Provides building blocks for generating synthetic networks with
multifractal error gating, optional BFS snapshot collection, and
parallel batch generation.  Higher-level orchestration (e.g. ranked
selection of the best candidates) lives in ``generate_select.py``.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from typing import List

import numpy as np

from analysis.error_checker import ErrorChecker, create_error_checker
from configs import BaseConfig, DatasetId
from graphs import GraphGenerator
from handlers import AttributesCalculator, Mapper, RunAgent, Saver
from utils import save_bfs_snapshot, trim_graph

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _build_temp_graph(positions, edges):
    """Build a lightweight SynthGraph from raw BFS callback data.

    Both *positions* and *edges* are already snapshot-time copies
    (positions is a new list, edges is a ``set()`` copy made inside
    ``_bfs_network``), so no shared-reference issues arise.
    """
    from graphs.synth_graph import SynthGraph

    pos_arr = np.array(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    edge_indices = []
    for p1, p2 in edges:
        u = pos_to_idx.get(p1)
        v = pos_to_idx.get(p2)
        if u is not None and v is not None:
            edge_indices.append([u, v])
    edge_arr = np.array(edge_indices) if edge_indices else np.empty((0, 2), dtype=int)
    return SynthGraph.from_edge_list(pos_arr, edge_arr).largest_connected_component()


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(
    exit_event,
    error_checker: ErrorChecker,
    attributes: AttributesCalculator,
    mapper: Mapper,
):
    if _should_exit(exit_event):
        return None, float("inf")

    generator = GraphGenerator(attributes)
    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        try:
            synthetic_graph = generator.generate_network()
            synthetic_graph = trim_graph(synthetic_graph, attributes.average_degree)
            mapper.assign_weights(synthetic_graph)

            if _should_exit(exit_event):
                return None, float("inf")

            passed, error_ = error_checker.check(synthetic_graph)
            if passed:
                return synthetic_graph, error_

        except KeyboardInterrupt:
            logger.info(SIGINT_INFO)
            raise
        except Exception as exc:
            logger.error(f"Exception occurred: {exc}. Retrying...", exc_info=True)

    logger.warning("Max attempts reached. Aborting!")
    return None, float("inf")


def _generate_single_network(
    exit_event,
    error_checker: ErrorChecker,
    attributes: AttributesCalculator,
    mapper: Mapper,
):
    """Generate one network with error gating and retry loop."""
    if exit_event.is_set():
        return None, float("inf")

    generator = GraphGenerator(attributes)
    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        try:
            graph = generator.generate_network()
            graph = trim_graph(graph, attributes.average_degree)
            mapper.assign_weights(graph)

            if exit_event.is_set():
                return None, float("inf")

            passed, error = error_checker.check(graph)
            if passed:
                return graph, error

            logger.debug(f"Attempt {attempt + 1}: error {error:.4f}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"Attempt {attempt + 1} failed: {exc}", exc_info=True)

    logger.warning("Max attempts reached. Returning None.")
    return None, float("inf")


def _generate_single_network_collecting_snapshots(
    exit_event,
    error_checker: ErrorChecker,
    attributes: AttributesCalculator,
    mapper: Mapper,
    snapshot_interval: int,
    early_check_node_count: int = 0,
):
    """Generate one network, collecting raw snapshot data in memory.

    Returns ``(graph, error, snapshots)`` where *snapshots* is a list
    of ``(positions, edges, frame, step_idx)`` tuples.  No rendering
    happens here — the caller decides which candidates are worth plotting.

    Parameters
    ----------
    early_check_node_count : int
        When > 0, run an error pre-check once the BFS reaches this many
        nodes (typically the original network's node count).  The partial
        network is trimmed and weighted so its features are comparable to
        the original's.  If the check fails, the BFS is aborted immediately.
        Has no effect on abort behavior when a ``NullErrorChecker`` is in
        use (it never fails); callers pass 0 to skip the pre-check entirely.
    """
    if exit_event.is_set():
        return None, float("inf"), []

    avg_degree = attributes.average_degree
    generator = GraphGenerator(attributes)
    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        snapshots = []
        early_aborted = False
        early_checked = False

        try:

            def on_snapshot(positions, edges, frame, step_idx):
                nonlocal early_aborted, early_checked
                snapshots.append((positions, edges, frame, step_idx))

                if (
                    early_check_node_count > 0
                    and not early_checked
                    and len(positions) >= early_check_node_count
                ):
                    early_checked = True
                    temp_graph = _build_temp_graph(positions, edges)
                    temp_graph = trim_graph(temp_graph, avg_degree)
                    mapper.assign_weights(temp_graph)
                    passed, error = error_checker.check(temp_graph)
                    if not passed:
                        logger.debug(
                            f"Attempt {attempt + 1}: early pre-check "
                            f"failed at {len(positions)} nodes "
                            f"(error {error:.4f})"
                        )
                        early_aborted = True
                        return False

            graph = generator.generate_network_with_snapshots(
                snapshot_callback=on_snapshot,
                snapshot_interval=snapshot_interval,
            )

            if early_aborted:
                continue

            graph = trim_graph(graph, attributes.average_degree)
            mapper.assign_weights(graph)

            if exit_event.is_set():
                return None, float("inf"), []

            passed, error = error_checker.check(graph)
            if passed:
                return graph, error, snapshots

            logger.debug(f"Attempt {attempt + 1}: error {error:.4f}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"Attempt {attempt + 1} failed: {exc}", exc_info=True)

    logger.warning("Max attempts reached. Returning None.")
    return None, float("inf"), []


# ------------------------------------------------------------------ #
# Batch generation (standard, no selection)
# ------------------------------------------------------------------ #


def compute_average_error(errors: List) -> float:
    valid_errors = [e for e in errors if not np.isinf(e)]
    if not valid_errors:
        return float("inf")

    mean_e, std_e = np.mean(valid_errors), np.std(valid_errors)
    non_outliers = [e for e in valid_errors if abs(e - mean_e) <= 2.0 * std_e]
    return round(np.mean(non_outliers), 3) if non_outliers else float("inf")


def generate_with_multiprocessing(data_agent: RunAgent):
    num_network, num_figures = (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER,
        BaseConfig.SYNTHETIC_GRAPH_NUMBER,
    )
    if num_network <= 0:
        logger.info("SYNTHETIC_NETWORK_NUMBER is 0 — skipping synthetic generation.")
        return

    errors, futures = [], []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    error_checker = create_error_checker()
    if BaseConfig.SYNTHETIC_NETWORK_NUMBER > 0:
        error_checker.compute_reference(data_agent.get_original_network())

    max_workers = BaseConfig.get_max_workers(num_network)
    logger.info(f"Using {max_workers} worker(s) for {num_network} networks")

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    error_checker,
                    data_agent.attributes,
                    data_agent.mapper,
                )
                for _ in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    break

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(f"({progress}%) Synthetic graph generated.")
                    next_log += 10

                if (num_figures := num_figures - 1) >= 0:
                    data_agent.save("synthetic_graph", content=synthetic_graph)
                data_agent.add_synthetic_graph(synthetic_graph)
                errors.append(error)
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        exit_event.set()
        for future in futures:
            future.cancel()

    avg_err = compute_average_error(errors)
    prefix = f"len_{len(errors)}_err_{avg_err:.3f}"
    data_agent.save_synthetic_outputs(prefix)

    if BaseConfig.FULL_ANALYSIS:
        data_agent.multifractal_analysis_in_generate_mode()
        data_agent.save("analysis_data")
        data_agent.save("analysis_figure")


# ------------------------------------------------------------------ #
# Single-network snapshot generation
# ------------------------------------------------------------------ #


def generate_with_snapshots(data_agent: RunAgent):
    """Generate a single network while saving intermediate BFS snapshots.

    Snapshot rendering is offloaded to a process pool so the BFS
    generation loop is never blocked by matplotlib / PNG I/O.
    """
    from concurrent.futures import ProcessPoolExecutor

    snapshot_dir = os.path.join(data_agent.saver.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    interval = BaseConfig.SNAPSHOT_INTERVAL
    style = getattr(BaseConfig, "PLOT_STYLE", {})

    plot_pool = ProcessPoolExecutor(max_workers=BaseConfig.get_snapshot_plot_workers())
    plot_futures = []

    def on_snapshot(positions, edges, frame, step_idx):
        fut = plot_pool.submit(
            save_bfs_snapshot,
            positions,
            edges,
            frame,
            step_idx,
            snapshot_dir,
            **style,
        )
        plot_futures.append(fut)

    generator = GraphGenerator(data_agent.attributes)
    synthetic_graph = generator.generate_network_with_snapshots(
        snapshot_callback=on_snapshot,
        snapshot_interval=interval,
    )
    synthetic_graph = trim_graph(synthetic_graph, data_agent.attributes.average_degree)
    data_agent.mapper.assign_weights(synthetic_graph)

    for fut in plot_futures:
        fut.result()
    plot_pool.shutdown(wait=False)
    logger.info(f"{len(plot_futures)} snapshots rendered")

    data_agent.save("synthetic_graph", content=synthetic_graph)
    data_agent.add_synthetic_graph(synthetic_graph)
    data_agent.save_synthetic_outputs(prefix="snapshot_run")

    logger.info(f"Snapshot generation complete. Snapshots saved to {snapshot_dir}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def run_for_dataset(dataset_id: DatasetId):
    """Process a single dataset identified by DatasetId."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(BaseConfig())
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_report")
    data_agent.save("original_graph")

    if BaseConfig.SNAPSHOT_INTERVAL > 0:
        generate_with_snapshots(data_agent)
    else:
        generate_with_multiprocessing(data_agent)


def main(config_cls=None):
    if config_cls is None:
        from configs.generate_mode import GenConfigSnapshot

        config_cls = GenConfigSnapshot
    config_cls.initialize()

    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
