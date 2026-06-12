# src/pipelines/sweep.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import numpy as np
import wandb

from analysis.error_checker import ErrorChecker, create_error_checker

# noinspection PyUnresolvedReferences
from configs import BaseConfig, DatasetId
from graphs import GraphGenerator
from graphs._graph_node import GraphNode
from handlers import RunAgent
from pipelines.generate import compute_average_error
from utils import build_graph, trim_graph

_CONFIG_MAP = {
    "A": "SweepConfigA",
    "B": "SweepConfigB",
    "C": "SweepConfigC",
    "D": "SweepConfigD",
}


def _init_config(config_key=None):
    """Resolve and initialize the sweep config. Call once before sweeping."""
    import configs.sweep_mode as sweep_mod

    name = _CONFIG_MAP.get(config_key, "SweepConfig")
    cfg = getattr(sweep_mod, name)
    cfg.initialize()
    BaseConfig.SYNTHETIC_NETWORK_NUMBER = 100
    BaseConfig.SYNTHETIC_GRAPH_NUMBER = 0
    BaseConfig.disable_saving("Sweeping Experiment")


EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"
DEFAULT_NF_RANGE = (0.3, 2.0)
DEFAULT_EF_RANGE = (0.3, 2.0)


def _build_factors(lo, hi, step=0.1):
    n = round((hi - lo) / step) + 1
    return [round(lo + step * i, 1) for i in range(n)]


logger = logging.getLogger(__name__)


def _generate_with_factors(
    exit_event, error_checker: ErrorChecker, attributes, mapper, nf, ef
):
    """Spawn-safe: uses the same BFS path as the hybrid tile generator."""
    import random

    BaseConfig.CLOSED_NODES_FACTOR = nf
    BaseConfig.CLOSED_EDGES_FACTOR = ef

    if exit_event.is_set():
        return None, float("inf")

    GraphNode.initialize(attributes)

    for attempt in range(BaseConfig.MAX_ATTEMPTS):
        seed = os.getpid() ^ attempt
        random.seed(seed)
        np.random.seed(seed % (2**31))

        try:
            result = GraphGenerator._bfs_network_with_frontier()
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


def generate_networks(data_agent, error_checker: ErrorChecker, nf, ef):
    """Generate networks with the given factors. Return (avg_error, success_rate)."""
    num_network = BaseConfig.SYNTHETIC_NETWORK_NUMBER
    exit_event = Manager().Event()
    max_workers = BaseConfig.get_max_workers(num_network)

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
                    nf,
                    ef,
                )
                for _ in range(num_network)
            ]
            next_log = 10
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    continue

                progress = round(idx / num_network * 100, 2)
                if progress >= next_log:
                    logger.info(f"({progress}%) Synthetic graph generated.")
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


def run_for_dataset(dataset_id: DatasetId, node_factors, edge_factors) -> None:
    """Create a wandb sweep for a single dataset and run all trials."""
    logger.info(f"Processing dataset: {dataset_id}")
    logger.info(
        f"  nf: {node_factors[0]}-{node_factors[-1]} ({len(node_factors)} values)"
    )
    logger.info(
        f"  ef: {edge_factors[0]}-{edge_factors[-1]} ({len(edge_factors)} values)"
    )
    logger.info(f"  total trials: {len(node_factors) * len(edge_factors)}")

    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    error_checker = create_error_checker()
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

            error, success_rate = generate_networks(data_agent, error_checker, nf, ef)
            if error is None:
                error = float("inf")

            wandb.log({"error": error, "success_rate": success_rate})

    sweep_id = wandb.sweep(sweep_config, project=EXPERIMENT_PROJECT_NAME)
    wandb.agent(sweep_id, function=trial)


def main(config=None, nf_range=None, ef_range=None):
    _init_config(config)

    nf_lo, nf_hi = nf_range or DEFAULT_NF_RANGE
    ef_lo, ef_hi = ef_range or DEFAULT_EF_RANGE
    node_factors = _build_factors(nf_lo, nf_hi)
    edge_factors = _build_factors(ef_lo, ef_hi)

    assert (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER != 0
    ), "Sweeping experiments require synthetic networks."

    wandb.login()
    try:
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id, node_factors, edge_factors)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
