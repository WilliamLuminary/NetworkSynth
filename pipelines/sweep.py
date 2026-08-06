# src/pipelines/sweep.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import wandb

from analysis.error_checker import ErrorChecker, create_error_checker

# noinspection PyUnresolvedReferences
from configs import DatasetId, SynthParams
from configs.base_config import tagged
from graphs import GraphGenerator
from graphs._graph_node import GraphNode
from handlers import RunAgent, attach_run_log, create_run_paths
from pipelines.generate import compute_average_error
from utils import apply_seed, build_graph, trim_graph


def _init_config():
    """Initialize the sweep config. Call once before sweeping.

    Sweeps every dataset the config lists; there is no per-dataset variant.
    """
    from configs.sweep_mode import SweepConfig as cfg

    cfg.initialize()
    cfg.SYNTHETIC_NETWORK_NUMBER = 100
    cfg.SYNTHETIC_GRAPH_NUMBER = 0
    cfg.disable_saving("Sweeping Experiment")
    return cfg


EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"
DEFAULT_NF_RANGE = (0.3, 2.0)
DEFAULT_EF_RANGE = (0.3, 2.0)


def _build_factors(lo, hi, step=0.1):
    n = round((hi - lo) / step) + 1
    return [round(lo + step * i, 1) for i in range(n)]


logger = logging.getLogger(__name__)


def _generate_with_factors(
    exit_event, error_checker: ErrorChecker, attributes, mapper, params
):
    """Uses the same BFS path as the hybrid tile generator.

    *params* already carries this trial's node/edge factors, derived in the
    parent - no global mutation, and nothing read from inherited class state.
    """
    if exit_event.is_set():
        return None, float("inf")

    GraphNode.initialize(attributes, params)

    for attempt in range(params.max_attempts):
        # Seeded from this trial's params rather than the PID, so a seeded
        # sweep repeats. Attempt index keeps retries from re-drawing the same
        # failed network.
        apply_seed(None if params.seed is None else params.seed + attempt)

        try:
            result = GraphGenerator._bfs_network_with_frontier(
                params.synthetic_frame_size
            )
            inner_nodes, inner_edges, *_ = result

            if not inner_nodes or len(inner_nodes) < 100:
                continue

            graph = build_graph(inner_nodes, inner_edges, arg_type="graph_node")
            graph = trim_graph(graph, attributes.average_degree)
            mapper.assign_weights(graph)

            if exit_event.is_set():
                return None, float("inf")

            passed, error = error_checker.check(graph)
            if passed:
                return graph, error
        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    return None, float("inf")


def generate_networks(data_agent, error_checker: ErrorChecker, nf, ef, config):
    """Generate networks with the given factors. Return (avg_error, success_rate)."""
    from dataclasses import replace

    num_network = config.SYNTHETIC_NETWORK_NUMBER
    exit_event = Manager().Event()
    max_workers = config.get_max_workers(num_network)

    # This trial's factors live in an immutable params object rather than being
    # written into global config, so trials cannot interfere with each other.
    trial_params = replace(
        SynthParams.from_config(config),
        closed_nodes_factor=nf,
        closed_edges_factor=ef,
    )

    errors = []
    futures = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _generate_with_factors,
                    exit_event,
                    error_checker,
                    data_agent.attributes,
                    data_agent.mapper,
                    trial_params.for_worker(i),
                )
                for i in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    continue

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(
                        f"({progress}%) Synthetic graph generated.",
                        extra=tagged("PROGRESS", percent=progress),
                    )
                    next_log += 10

                errors.append(error)
    except KeyboardInterrupt:
        logger.info("SIGINT received. Terminating child processes...")
        raise
    finally:
        exit_event.set()
        for f in futures:
            f.cancel()

    success_rate = len(errors) / num_network if num_network > 0 else 0.0
    avg_error = compute_average_error(errors)
    return avg_error, success_rate


def run_for_dataset(
    dataset_id: DatasetId, node_factors, edge_factors, config, run_paths
) -> None:
    """Create a wandb sweep for a single dataset and run all trials."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(
        f"  nf: {node_factors[0]}-{node_factors[-1]} ({len(node_factors)} values)"
    )
    logger.info(
        f"  ef: {edge_factors[0]}-{edge_factors[-1]} ({len(edge_factors)} values)"
    )
    logger.info(f"  total trials: {len(node_factors) * len(edge_factors)}")

    data_agent = RunAgent(config, run_paths=run_paths, dataset_id=dataset_id)
    data_agent.prepare_data()
    error_checker = create_error_checker(config)
    error_checker.compute_reference(data_agent.get_original_network())

    sweep_config = {
        "name": f"sweep-{dataset_id}",
        "method": "grid",
        "metric": {"name": "error", "goal": "minimize"},
        "parameters": {
            "node_factor": {"values": node_factors},
            "edge_factor": {"values": edge_factors},
        },
    }

    def trial():
        with wandb.init():
            nf = wandb.config.node_factor
            ef = wandb.config.edge_factor
            logger.info(f"Trial: nf={nf}, ef={ef}")

            error, success_rate = generate_networks(
                data_agent, error_checker, nf, ef, config
            )
            if error is None:
                error = float("inf")

            wandb.log({"error": error, "success_rate": success_rate})

    sweep_id = wandb.sweep(sweep_config, project=EXPERIMENT_PROJECT_NAME)
    wandb.agent(sweep_id, function=trial)


def main(nf_range=None, ef_range=None):
    cfg = _init_config()

    # CLI argument wins, then the config's range, then the module default.
    nf_lo, nf_hi = nf_range or getattr(cfg, "NF_RANGE", DEFAULT_NF_RANGE)
    ef_lo, ef_hi = ef_range or getattr(cfg, "EF_RANGE", DEFAULT_EF_RANGE)
    node_factors = _build_factors(nf_lo, nf_hi)
    edge_factors = _build_factors(ef_lo, ef_hi)

    assert (
        cfg.SYNTHETIC_NETWORK_NUMBER != 0
    ), "Sweeping experiments require synthetic networks."

    wandb.login()
    try:
        run_paths = create_run_paths(cfg)
        attach_run_log(run_paths.root, cfg.RUN_ID)
        for dataset_id in cfg.get_datasets():
            run_for_dataset(dataset_id, node_factors, edge_factors, cfg, run_paths)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        # Re-raised so the entry point can exit non-zero: a cancelled
        # run must not look like a completed one to a caller.
        raise


if __name__ == "__main__":
    main()
