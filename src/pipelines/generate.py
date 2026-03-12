# src/pipelines/generate.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from typing import List

import numpy as np

from analysis import MultifractalAnalyzer
from configs import BaseConfig, DatasetId
from configs.generate_mode import GenConfigSnapshot as GenConfig
from graphs import GraphGenerator
from handlers import AttributesCalculator, Mapper, RunAgent, Saver
from utils import (
    compute_network_metrics,
    metric_distance,
    save_bfs_snapshot,
    trim_graph,
)

GenConfig.initialize()
# BaseConfig.disable_saving("debug")

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def _should_exit(exit_event) -> bool:
    if exit_event.is_set():
        logger.info(SIGINT_INFO)
        return True
    return False


def generate_synthetic_network(
    exit_event, std_err_fea, attributes: AttributesCalculator, mapper: Mapper
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

            err_fea = MultifractalAnalyzer(synthetic_graph).analyze_error_features()
            error_ = MultifractalAnalyzer.analyze_error(err_fea, std_err_fea)
            if error_ < BaseConfig.ERROR_TOLERANCE:
                return synthetic_graph, error_

        except KeyboardInterrupt:
            logger.info(SIGINT_INFO)
            raise
        except Exception as exc:
            logger.error(f"Exception occurred: {exc}. Retrying...", exc_info=True)

    logger.warning("Max attempts reached. Aborting!")
    return None, float("inf")


def generate_with_multiprocessing(data_agent: RunAgent):
    num_network, num_figures = (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER,
        BaseConfig.SYNTHETIC_GRAPH_NUMBER,
    )
    errors, futures = [], []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    std_err_fea = (
        None
        if BaseConfig.SYNTHETIC_NETWORK_NUMBER == 0
        else MultifractalAnalyzer(
            data_agent.get_original_network()
        ).analyze_error_features()
    )

    max_workers = BaseConfig.get_max_workers(num_network)
    logger.info(f"Using {max_workers} worker(s) for {num_network} networks")

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    std_err_fea,
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


def compute_average_error(errors: List) -> float:
    valid_errors = [e for e in errors if not np.isinf(e)]
    if not valid_errors:
        return float("inf")

    mean_e, std_e = np.mean(valid_errors), np.std(valid_errors)
    non_outliers = [e for e in valid_errors if abs(e - mean_e) <= 2.0 * std_e]
    return round(np.mean(non_outliers), 3) if non_outliers else float("inf")


def _generate_single_network(
    exit_event, attributes: AttributesCalculator, mapper: Mapper
):
    """Generate one network without multifractal quality gating."""
    if exit_event.is_set():
        return None
    generator = GraphGenerator(attributes)
    try:
        graph = generator.generate_network()
        graph = trim_graph(graph, attributes.average_degree)
        mapper.assign_weights(graph)
        return graph
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.error(f"Generation failed: {exc}", exc_info=True)
        return None


def _generate_single_network_with_snapshots(
    exit_event,
    attributes: AttributesCalculator,
    mapper: Mapper,
    snapshot_dir: str,
    snapshot_interval: int,
):
    """Generate one network with BFS snapshots saved to *snapshot_dir*."""
    if exit_event.is_set():
        return None
    os.makedirs(snapshot_dir, exist_ok=True)

    def on_snapshot(positions, edges, frame, step_idx):
        save_bfs_snapshot(positions, edges, frame, step_idx, snapshot_dir)

    generator = GraphGenerator(attributes)
    try:
        graph = generator.generate_network_with_snapshots(
            snapshot_callback=on_snapshot,
            snapshot_interval=snapshot_interval,
        )
        graph = trim_graph(graph, attributes.average_degree)
        mapper.assign_weights(graph)
        return graph
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.error(f"Generation failed: {exc}", exc_info=True)
        return None


def generate_and_select(data_agent: RunAgent):
    """Generate many networks, rank by metric distance to original, save best.

    When ``SNAPSHOT_INTERVAL > 0``, each candidate also gets BFS
    snapshots under ``candidates/network_XX/snapshots/``.  After
    ranking, directories of non-best candidates are deleted.
    """
    import shutil

    num_network = BaseConfig.SYNTHETIC_NETWORK_NUMBER
    select_best = BaseConfig.SELECT_BEST
    snapshot_interval = BaseConfig.SNAPSHOT_INTERVAL
    use_snapshots = snapshot_interval > 0

    ref_metrics = compute_network_metrics(data_agent.get_original_network())
    logger.info("Original metrics: %s", _fmt_metrics(ref_metrics))

    candidates_dir = os.path.join(data_agent.saver.output_dir, "candidates")

    graphs = []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    max_workers = BaseConfig.get_max_workers(num_network)
    logger.info(
        f"Generating {num_network} networks with {max_workers} worker(s)"
        + (f", snapshots every {snapshot_interval} nodes" if use_snapshots else "")
    )
    futures = []

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i in range(num_network):
                if use_snapshots:
                    snap_dir = os.path.join(
                        candidates_dir, f"network_{i:02d}", "snapshots"
                    )
                    future = executor.submit(
                        _generate_single_network_with_snapshots,
                        exit_event,
                        data_agent.attributes,
                        data_agent.mapper,
                        snap_dir,
                        snapshot_interval,
                    )
                else:
                    future = executor.submit(
                        _generate_single_network,
                        exit_event,
                        data_agent.attributes,
                        data_agent.mapper,
                    )
                future_to_idx[future] = i
                futures.append(future)

            for done_count, future in enumerate(as_completed(futures), start=1):
                original_idx = future_to_idx[future]
                graph = future.result()
                if graph is not None:
                    graphs.append((original_idx, graph))
                if done_count % max(1, num_network // 10) == 0:
                    logger.info(
                        f"({done_count}/{num_network}) " f"{len(graphs)} valid so far"
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

    logger.info(f"Evaluating metrics on {len(graphs)} networks ...")
    ranked = []
    for original_idx, g in graphs:
        m = compute_network_metrics(g)
        d = metric_distance(m, ref_metrics)
        ranked.append((d, original_idx, g, m))
    ranked.sort(key=lambda x: x[0])

    _log_ranking_table(ranked, ref_metrics)

    best = ranked[:select_best]
    best_indices = {idx for _, idx, _, _ in best}

    for _, _, graph, _ in best:
        data_agent.save("synthetic_graph", content=graph)
        data_agent.add_synthetic_graph(graph)

    prefix = f"best{len(best)}_of_{len(graphs)}"
    data_agent.save_synthetic_outputs(prefix)

    # Clean up non-best candidate directories
    if use_snapshots:
        all_indices = {idx for idx, _ in graphs}
        removed = 0
        for idx in all_indices - best_indices:
            network_dir = os.path.join(candidates_dir, f"network_{idx:02d}")
            if os.path.isdir(network_dir):
                shutil.rmtree(network_dir)
                removed += 1
        logger.info(f"Kept {len(best_indices)} candidate dirs, removed {removed}")

    report_path = os.path.join(data_agent.saver.output_dir, "metric_report.csv")
    _save_metric_report(ranked, ref_metrics, report_path)
    logger.info(
        f"Saved {len(best)} best networks (of {len(graphs)} total). "
        f"Report: {report_path}"
    )


def _fmt_metrics(m: dict) -> str:
    return (
        f"nodes={m['node_count']}  avg_deg={m['avg_degree']:.2f}  "
        f"clust={m['avg_clustering']:.4f}  length={m['avg_length']:.2f}  "
        f"angle={m['avg_angle']:.2f}"
    )


def _log_ranking_table(ranked, ref_metrics):
    hdr = (
        f"{'Rank':>4} {'Dist':>8} {'Nodes':>7} {'AvgDeg':>7} "
        f"{'Clust':>8} {'Length':>8} {'Angle':>8}"
    )
    sep = "-" * len(hdr)
    lines = ["\n" + sep, hdr, sep]
    for rank, (dist, _, _, m) in enumerate(ranked, 1):
        lines.append(
            f"{rank:4d} {dist:8.4f} {m['node_count']:7d} {m['avg_degree']:7.2f} "
            f"{m['avg_clustering']:8.4f} {m['avg_length']:8.2f} {m['avg_angle']:8.2f}"
        )
    lines.append(sep)
    r = ref_metrics
    lines.append(
        f"{'Orig':>4} {'':>8} {r['node_count']:7d} {r['avg_degree']:7.2f} "
        f"{r['avg_clustering']:8.4f} {r['avg_length']:8.2f} {r['avg_angle']:8.2f}"
    )
    lines.append(sep)
    logger.info("\n".join(lines))


def _save_metric_report(ranked, ref_metrics, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "distance",
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
                ref_metrics["node_count"],
                ref_metrics["avg_degree"],
                ref_metrics["avg_clustering"],
                ref_metrics["avg_length"],
                ref_metrics["avg_angle"],
            ]
        )
        for rank, (dist, _, _, m) in enumerate(ranked, 1):
            w.writerow(
                [
                    rank,
                    f"{dist:.6f}",
                    m["node_count"],
                    f"{m['avg_degree']:.4f}",
                    f"{m['avg_clustering']:.6f}",
                    f"{m['avg_length']:.4f}",
                    f"{m['avg_angle']:.4f}",
                ]
            )


def generate_with_snapshots(data_agent: RunAgent):
    """Generate a single network while saving intermediate BFS snapshots."""
    snapshot_dir = os.path.join(data_agent.saver.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    interval = BaseConfig.SNAPSHOT_INTERVAL

    def on_snapshot(positions, edges, frame, step_idx):
        save_bfs_snapshot(positions, edges, frame, step_idx, snapshot_dir)

    generator = GraphGenerator(data_agent.attributes)
    synthetic_graph = generator.generate_network_with_snapshots(
        snapshot_callback=on_snapshot,
        snapshot_interval=interval,
    )
    synthetic_graph = trim_graph(synthetic_graph, data_agent.attributes.average_degree)
    data_agent.mapper.assign_weights(synthetic_graph)

    data_agent.save("synthetic_graph", content=synthetic_graph)
    data_agent.add_synthetic_graph(synthetic_graph)
    data_agent.save_synthetic_outputs(prefix="snapshot_run")

    logger.info(f"Snapshot generation complete. Snapshots saved to {snapshot_dir}")


def run_for_dataset(dataset_id: DatasetId):
    """Process a single dataset identified by DatasetId."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(BaseConfig())
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_graph")

    if BaseConfig.SELECT_BEST > 0:
        generate_and_select(data_agent)
    elif BaseConfig.SNAPSHOT_INTERVAL > 0:
        generate_with_snapshots(data_agent)
    else:
        generate_with_multiprocessing(data_agent)


def main():
    try:
        Saver.initialize()
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
