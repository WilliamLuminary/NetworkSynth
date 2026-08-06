# src/pipelines/generate_select.py
"""
Generate-and-select pipeline — generate N candidate networks, rank by
5 metrics against the original, and keep the top K.

Reuses the core generation workers from ``generate.py``.  When
``SNAPSHOT_INTERVAL > 0``, raw BFS snapshot data is collected in
memory during generation (zero rendering cost).  Only the winning
candidates get their snapshots rendered to PNGs after ranking.
"""
import csv
import logging
import os

from analysis.error_checker import NullErrorChecker, create_error_checker
from configs import DatasetId, SynthParams
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    RunAgent,
    attach_run_log,
    create_run_paths,
    write_manifest,
)
from pipelines.generate import (
    SIGINT_INFO,
    _generate_single_network,
    _generate_single_network_collecting_snapshots,
)
from utils import compute_network_metrics, metric_distance, save_bfs_snapshot

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Snapshot rendering (only called for winners)
# ------------------------------------------------------------------ #


def _render_snapshots(snapshot_data, output_dir, style, plot_workers: int):
    """Render collected snapshot data to PNGs using a process pool."""
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(output_dir, exist_ok=True)
    if not snapshot_data:
        return

    with ProcessPoolExecutor(max_workers=min(plot_workers, len(snapshot_data))) as pool:
        futs = [
            pool.submit(
                save_bfs_snapshot,
                positions,
                edges,
                frame,
                step_idx,
                output_dir,
                **style,
            )
            for positions, edges, frame, step_idx in snapshot_data
        ]
        for fut in futs:
            fut.result()

    logger.info(f"Rendered {len(snapshot_data)} snapshots to {output_dir}")


# ------------------------------------------------------------------ #
# Reporting helpers
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
# Main orchestrator
# ------------------------------------------------------------------ #


def generate_and_select(data_agent: RunAgent, config):
    """Generate many networks, rank by metric distance to original, save best.

    Each candidate is validated with the multifractal error check.
    Workers retry up to ``MAX_ATTEMPTS`` times until a network passes
    ``ERROR_TOLERANCE``.

    When ``SNAPSHOT_INTERVAL > 0``, raw snapshot data is collected in
    memory during BFS (no rendering).  Only the best candidates get
    their snapshots rendered to PNGs after ranking.
    """
    num_network = config.SYNTHETIC_NETWORK_NUMBER
    select_best = config.SELECT_BEST
    snapshot_interval = config.SNAPSHOT_INTERVAL
    snapshot_style = getattr(config, "PLOT_STYLE", {})
    use_snapshots = snapshot_interval > 0

    original_network = data_agent.get_original_network()
    original_node_count = original_network.number_of_nodes()
    ref_metrics = compute_network_metrics(original_network)
    error_checker = create_error_checker(config)
    skip_mf = isinstance(error_checker, NullErrorChecker)
    error_checker.compute_reference(original_network)
    logger.info("Original metrics: %s", _fmt_metrics(ref_metrics))

    candidates_dir = os.path.join(data_agent.saver.output_dir, "candidates")

    graphs = []
    from multiprocessing import Manager

    exit_event = Manager().Event()
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

    # Built in the parent so workers get their parameters explicitly rather
    # than inheriting a mutated BaseConfig (which only works on `fork`).
    params = SynthParams.from_config(config)

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i in range(num_network):
                if use_snapshots:
                    future = executor.submit(
                        _generate_single_network_collecting_snapshots,
                        exit_event,
                        error_checker,
                        data_agent.attributes,
                        data_agent.mapper,
                        snapshot_interval,
                        early_node_count,
                        params.for_worker(i),
                    )
                else:
                    future = executor.submit(
                        _generate_single_network,
                        exit_event,
                        error_checker,
                        data_agent.attributes,
                        data_agent.mapper,
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
        data_agent.save("synthetic_graph", content=graph)
        data_agent.add_synthetic_graph(graph)

    prefix = f"best{len(best)}_of_{len(graphs)}"
    data_agent.save_synthetic_outputs(prefix)

    if use_snapshots:
        logger.info(f"Rendering snapshots for {len(best)} best candidates ...")
        for rank, (_, orig_idx, _, _, _, snaps) in enumerate(best, start=1):
            snap_dir = os.path.join(
                candidates_dir,
                f"rank{rank:02d}_network_{orig_idx:02d}",
                "snapshots",
            )
            _render_snapshots(snaps, snap_dir, snapshot_style)

    report_path = os.path.join(data_agent.saver.output_dir, "metric_report.csv")
    _save_metric_report(ranked, ref_metrics, report_path)
    logger.info(
        f"Saved {len(best)} best networks (of {len(graphs)} total). "
        f"Report: {report_path}"
    )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def run_for_dataset(dataset_id: DatasetId, config, run_paths):
    """Process a single dataset: prepare data, then generate-and-select."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(config())
    data_agent = RunAgent(config, run_paths=run_paths, dataset_id=dataset_id)
    data_agent.prepare_data()
    data_agent.save("original_image")
    data_agent.save("original_network")
    data_agent.save("original_property")
    data_agent.save("original_report")
    data_agent.save("original_graph")

    generate_and_select(data_agent, config)


def main(config_cls=None):
    if config_cls is None:
        from configs.generate_mode import GenConfigSnapshot

        config_cls = GenConfigSnapshot
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
