# SPDX-License-Identifier: GPL-3.0-or-later
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import logging
from typing import List

import numpy as np

from analysis.error_checker import (
    ErrorChecker,
    NullErrorChecker,
    create_error_checker,
)
from configs import DatasetId, SynthParams
from graphs import GraphGenerator
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
    compute_network_metrics,
    metric_distance,
    save_bfs_snapshot,
    spawn_context,
    tagged,
    trim_graph,
)

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _build_temp_graph(positions, edges):
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
    params: SynthParams,
):

    if _should_exit(exit_event):
        return None, float("inf")

    apply_seed(params.seed)
    generator = GraphGenerator(attributes, params)
    for attempt in range(params.max_attempts):
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
    params: SynthParams,
):

    if exit_event.is_set():
        return None, float("inf")

    apply_seed(params.seed)
    generator = GraphGenerator(attributes, params)
    for attempt in range(params.max_attempts):
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
    early_check_node_count: int,
    params: SynthParams,
):

    if exit_event.is_set():
        return None, float("inf"), []

    apply_seed(params.seed)
    avg_degree = attributes.average_degree
    generator = GraphGenerator(attributes, params)
    for attempt in range(params.max_attempts):
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
                snapshot_round_interval=snapshot_interval,
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


def compute_average_error(errors: List) -> float:
    valid_errors = [e for e in errors if not np.isinf(e)]
    if not valid_errors:
        return float("inf")

    mean_e, std_e = np.mean(valid_errors), np.std(valid_errors)
    non_outliers = [e for e in valid_errors if abs(e - mean_e) <= 2.0 * std_e]
    return round(np.mean(non_outliers), 3) if non_outliers else float("inf")


def generate_with_multiprocessing(run: GenerationRun, config):
    num_network, num_figures = (
        config.SYNTHETIC_NETWORK_NUMBER,
        config.SYNTHETIC_GRAPH_NUMBER,
    )
    if num_network <= 0:
        logger.info("SYNTHETIC_NETWORK_NUMBER is 0 — skipping synthetic generation.")
        return

    errors, futures = [], []

    exit_event = spawn_context().Manager().Event()
    error_checker = create_error_checker(config)
    if config.SYNTHETIC_NETWORK_NUMBER > 0:
        error_checker.compute_reference(run.original)

    max_workers = config.get_max_workers(num_network)
    logger.info(f"Using {max_workers} worker(s) for {num_network} networks")

    params = SynthParams.from_config(config)

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_context()
        ) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    error_checker,
                    run.attributes,
                    run.mapper,
                    params.for_worker(i),
                )
                for i in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    break

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(
                        f"({progress}%) Synthetic graph generated.",
                        extra=tagged("PROGRESS", percent=progress),
                    )
                    next_log += 10

                if (num_figures := num_figures - 1) >= 0:
                    run.save_synthetic_plot(synthetic_graph)
                run.add_synthetic_graph(synthetic_graph)
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
    run.save_synthetic_outputs(prefix)


def generate_with_snapshots(run: GenerationRun, config):
    from concurrent.futures import ProcessPoolExecutor

    snapshot_dir = os.path.join(run.saver.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    interval = config.SNAPSHOT_INTERVAL
    style = config.render("bfs_snapshot")

    plot_pool = ProcessPoolExecutor(
        max_workers=config.get_snapshot_plot_workers(), mp_context=spawn_context()
    )
    plot_futures = []

    def on_snapshot(positions, edges, frame, step_idx):
        fut = plot_pool.submit(
            save_bfs_snapshot,
            positions,
            edges,
            frame,
            step_idx,
            snapshot_dir,
            style,
            config.snapshot_formats(),
        )
        plot_futures.append(fut)

    params = SynthParams.from_config(config)
    apply_seed(params.seed)
    generator = GraphGenerator(run.attributes, params)
    synthetic_graph = generator.generate_network_with_snapshots(
        snapshot_callback=on_snapshot,
        snapshot_round_interval=interval,
    )
    synthetic_graph = trim_graph(synthetic_graph, run.attributes.average_degree)
    run.mapper.assign_weights(synthetic_graph)

    for fut in plot_futures:
        fut.result()
    plot_pool.shutdown(wait=False)
    logger.info(f"{len(plot_futures)} snapshots rendered")

    run.save_synthetic_plot(synthetic_graph)
    run.add_synthetic_graph(synthetic_graph)
    run.save_synthetic_outputs(prefix="snapshot_run")

    logger.info(f"Snapshot generation complete. Snapshots saved to {snapshot_dir}")


def _render_snapshots(snapshot_data, output_dir, style, formats, plot_workers: int):
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(output_dir, exist_ok=True)
    if not snapshot_data:
        return

    with ProcessPoolExecutor(
        max_workers=min(plot_workers, len(snapshot_data)),
        mp_context=spawn_context(),
    ) as pool:
        futs = [
            pool.submit(
                save_bfs_snapshot,
                positions,
                edges,
                frame,
                step_idx,
                output_dir,
                style,
                formats,
            )
            for positions, edges, frame, step_idx in snapshot_data
        ]
        for fut in futs:
            fut.result()

    logger.info(f"Rendered {len(snapshot_data)} snapshots to {output_dir}")


def _fmt_metrics(m: dict) -> str:
    return (
        f"nodes={m['node_count']}  avg_deg={m['avg_degree']:.2f}  "
        f"clust={m['avg_clustering']:.4f}  length={m['avg_length']:.2f}  "
        f"angle={m['avg_angle']:.2f}"
    )


def _log_ranking_table(ranked, ref_metrics):
    hdr = (
        f"{'Rank':>4} {'Dist':>8} {'MFErr':>8} {'Nodes':>7} {'AvgDeg':>7} "
        f"{'Clust':>8} {'Length':>8} {'Angle':>8}"
    )
    sep = "-" * len(hdr)
    lines = ["\n" + sep, hdr, sep]
    for rank, (dist, _, _, m, mf_err, *_rest) in enumerate(ranked, 1):
        lines.append(
            f"{rank:4d} {dist:8.4f} {mf_err:8.4f} "
            f"{m['node_count']:7d} {m['avg_degree']:7.2f} "
            f"{m['avg_clustering']:8.4f} {m['avg_length']:8.2f} {m['avg_angle']:8.2f}"
        )
    lines.append(sep)
    r = ref_metrics
    lines.append(
        f"{'Orig':>4} {'':>8} {'':>8} {r['node_count']:7d} {r['avg_degree']:7.2f} "
        f"{r['avg_clustering']:8.4f} {r['avg_length']:8.2f} {r['avg_angle']:8.2f}"
    )
    lines.append(sep)
    logger.info("\n".join(lines))


def _save_metric_report(ranked, ref_metrics, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "metric_distance",
                "multifractal_error",
                "node_count",
                "avg_degree",
                "avg_clustering",
                "avg_length",
                "avg_angle",
            ]
        )
        w.writerow(
            [
                "original",
                "",
                "",
                ref_metrics["node_count"],
                ref_metrics["avg_degree"],
                ref_metrics["avg_clustering"],
                ref_metrics["avg_length"],
                ref_metrics["avg_angle"],
            ]
        )
        for rank, (dist, _, _, m, mf_err, *_rest) in enumerate(ranked, 1):
            w.writerow(
                [
                    rank,
                    f"{dist:.6f}",
                    f"{mf_err:.6f}",
                    m["node_count"],
                    f"{m['avg_degree']:.4f}",
                    f"{m['avg_clustering']:.6f}",
                    f"{m['avg_length']:.4f}",
                    f"{m['avg_angle']:.4f}",
                ]
            )


def generate_and_select(run: GenerationRun, config):
    assert config.SELECT_BEST > 0, "generate_and_select needs SELECT_BEST > 0"

    num_network = config.SYNTHETIC_NETWORK_NUMBER
    select_best = config.SELECT_BEST
    snapshot_interval = config.SNAPSHOT_INTERVAL
    snapshot_style = config.render("bfs_snapshot")
    use_snapshots = snapshot_interval > 0

    original_network = run.original
    original_node_count = original_network.number_of_nodes()
    ref_metrics = compute_network_metrics(original_network)
    error_checker = create_error_checker(config)
    skip_mf = isinstance(error_checker, NullErrorChecker)
    error_checker.compute_reference(original_network)
    logger.info("Original metrics: %s", _fmt_metrics(ref_metrics))

    candidates_dir = os.path.join(run.saver.output_dir, "candidates")

    graphs = []

    exit_event = spawn_context().Manager().Event()
    max_workers = config.get_max_workers(num_network)
    early_node_count = original_node_count if use_snapshots and not skip_mf else 0
    logger.info(
        f"Generating {num_network} networks with {max_workers} worker(s), "
        f"selecting top {select_best}, "
        + (
            f"error_tolerance={config.ERROR_TOLERANCE}, "
            f"max_attempts={config.MAX_ATTEMPTS}"
            if not skip_mf
            else "MF error check DISABLED, "
        )
        + (f"snapshots every {snapshot_interval} nodes" if use_snapshots else "")
        + (f", early MF check at ~{early_node_count} nodes" if early_node_count else "")
    )
    futures = []

    params = SynthParams.from_config(config)

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_context()
        ) as executor:
            future_to_idx = {}
            for i in range(num_network):
                if use_snapshots:
                    future = executor.submit(
                        _generate_single_network_collecting_snapshots,
                        exit_event,
                        error_checker,
                        run.attributes,
                        run.mapper,
                        snapshot_interval,
                        early_node_count,
                        params.for_worker(i),
                    )
                else:
                    future = executor.submit(
                        _generate_single_network,
                        exit_event,
                        error_checker,
                        run.attributes,
                        run.mapper,
                        params.for_worker(i),
                    )
                future_to_idx[future] = i
                futures.append(future)

            for done_count, future in enumerate(as_completed(futures), start=1):
                original_idx = future_to_idx[future]
                if use_snapshots:
                    graph, mf_error, snaps = future.result()
                else:
                    graph, mf_error = future.result()
                    snaps = []
                if graph is not None:
                    graphs.append((original_idx, graph, mf_error, snaps))
                else:
                    logger.warning(
                        f"Network {original_idx:02d} failed after "
                        f"{config.MAX_ATTEMPTS} attempts"
                    )
                if done_count % max(1, num_network // 10) == 0:
                    logger.info(
                        f"({done_count}/{num_network}) {len(graphs)} valid so far"
                    )
    except KeyboardInterrupt:
        logger.info(SIGINT_INFO)
        raise
    finally:
        exit_event.set()
        for f in futures:
            f.cancel()

    if not graphs:
        logger.error("No valid networks generated.")
        return

    logger.info(
        f"{len(graphs)}/{num_network} networks passed error tolerance "
        f"({config.ERROR_TOLERANCE})"
    )

    logger.info(f"Evaluating metrics on {len(graphs)} networks ...")
    ranked = []
    for original_idx, g, mf_error, snaps in graphs:
        m = compute_network_metrics(g)
        d = metric_distance(m, ref_metrics)
        ranked.append((d, original_idx, g, m, mf_error, snaps))
    ranked.sort(key=lambda x: x[0])

    _log_ranking_table(ranked, ref_metrics)

    best = ranked[:select_best]

    for _, _, graph, _, _, _ in best:
        run.save_synthetic_plot(graph)
        run.add_synthetic_graph(graph)

    prefix = f"best{len(best)}_of_{len(graphs)}"
    run.save_synthetic_outputs(prefix)

    if use_snapshots:
        logger.info(f"Rendering snapshots for {len(best)} best candidates ...")
        for rank, (_, orig_idx, _, _, _, snaps) in enumerate(best, start=1):
            snap_dir = os.path.join(
                candidates_dir,
                f"rank{rank:02d}_network_{orig_idx:02d}",
                "snapshots",
            )
            _render_snapshots(
                snaps,
                snap_dir,
                snapshot_style,
                config.snapshot_formats(),
                config.get_snapshot_plot_workers(),
            )

    report_path = os.path.join(run.saver.output_dir, "metric_report.csv")
    _save_metric_report(ranked, ref_metrics, report_path)
    logger.info(
        f"Saved {len(best)} best networks (of {len(graphs)} total). "
        f"Report: {report_path}"
    )


def run_for_dataset(dataset_id: DatasetId, config, run_paths):
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(config())
    run = GenerationRun(config, run_paths, dataset_id)
    run.save_original()
    run.save(run.original_report(), "original_report")

    if config.SELECT_BEST > 0:
        generate_and_select(run, config)
    # Snapshots replace the batch with a single network rather than adding to
    # it: the run produces one network, saved under a snapshot_run prefix.
    elif config.SNAPSHOT_INTERVAL > 0:
        generate_with_snapshots(run, config)
    else:
        generate_with_multiprocessing(run, config)


def main(config_cls=None):
    if config_cls is None:
        from configs.generate_mode.config_snapshot import SnapshotConfig

        config_cls = SnapshotConfig
    config_cls.initialize()

    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            run_for_dataset(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not config_cls.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
