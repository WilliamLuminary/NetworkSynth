# src/scripts/sweep_tmp.py
"""
Hyperparameter sweep over (CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR).

Generates 100 synthetic networks per (nf, ef) pair using multiprocessing,
computes the average multifractal error and success rate, and logs all
results to Weights & Biases (wandb) including a heatmap artifact.

Grid searched: nf and ef each from 0.4 to 2.0 in steps of 0.1
               → 17 × 17 = 289 parameter pairs.

Usage (from src/):
    python scripts/sweep_tmp.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Tuple

import wandb
from analysis import MultifractalAnalyzer

# noinspection PyUnresolvedReferences
from config import BaseConfig, DatasetId
from config.generate_mode import GenConfigTmp as GenConfig
from handlers import RunAgent, Saver
from pipelines.generate import compute_average_error, generate_synthetic_network

GenConfig.initialize()
BaseConfig.SYNTHETIC_NETWORK_NUMBER = 100
BaseConfig.SYNTHETIC_GRAPH_NUMBER = 0
BaseConfig.disable_saving("Sweeping Experiment")
Saver.initialize()

EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"
EXPERIMENT_NAME = "mosaic-sample-sweep"
NODE_FACTORS = [round(0.4 + 0.1 * i, 1) for i in range(17)]  # 0.4 … 2.0
EDGE_FACTORS = [round(0.4 + 0.1 * i, 1) for i in range(17)]  # 0.4 … 2.0

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def generate_networks_multiprocess(
    data_agent: RunAgent, std_err_fea
) -> Tuple[float, float]:
    """Return (average_error, success_rate)."""
    num_network = BaseConfig.SYNTHETIC_NETWORK_NUMBER
    errors, futures = [], []
    from multiprocessing import Manager

    exit_event = Manager().Event()
    max_workers = BaseConfig.get_max_workers(num_network)
    logger.info(
        f"Spawning pool with {max_workers} workers for {num_network} networks …"
    )
    try:
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
            next_log = 0
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    continue

                if (progress := round(idx / num_network * 100, 2)) >= (
                    next_log := next_log + 10
                ):
                    logger.info(f"({progress}%) Synthetic graph generated.")

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
        executor.shutdown(wait=True, cancel_futures=True)

    success_rate = len(errors) / num_network if num_network > 0 else 0.0
    avg_error = compute_average_error(errors)
    return avg_error, success_rate


def run_with_params(data_agent, ef, nf, std_err_fea):
    """Apply a single (nf, ef) pair and generate all networks for it."""
    BaseConfig.set_node_factor(nf)
    BaseConfig.set_edge_factor(ef)
    logger.info(BaseConfig())
    return generate_networks_multiprocess(data_agent, std_err_fea)


def run_for_dataset(dataset_id: DatasetId) -> None:
    """Process a single dataset identified by DatasetId."""
    logger.info(f"Processing dataset: {dataset_id}")
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    std_err_fea = MultifractalAnalyzer(
        data_agent.get_original_network()
    ).analyze_error_features()

    table = wandb.Table(columns=["node_factor", "edge_factor", "error", "success_rate"])
    for nf, ef in product(NODE_FACTORS, EDGE_FACTORS):
        error, success_rate = run_with_params(data_agent, ef, nf, std_err_fea)
        if error is None:
            error = float("inf")
        table.add_data(nf, ef, error, success_rate)
        wandb.log(
            {
                "node_factor": nf,
                "edge_factor": ef,
                "error": error,
                "success_rate": success_rate,
            }
        )

    wandb.log(
        {
            "my_heatmap": wandb.plot_table(
                "heatmap",
                table,
                {"x": "node_factor", "y": "edge_factor", "value": "error"},
            )
        }
    )


def main():
    assert (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER != 0
    ), "Sweeping experiments require synthetic networks."

    max_workers = BaseConfig.get_max_workers(BaseConfig.SYNTHETIC_NETWORK_NUMBER)
    logger.info(
        f"Using {max_workers} worker(s) for {BaseConfig.SYNTHETIC_NETWORK_NUMBER} networks per param pair"
    )

    wandb.login()
    wandb.init(
        project=EXPERIMENT_PROJECT_NAME,
        name=EXPERIMENT_NAME,
        dir=BaseConfig.PROJECT_ROOT,
    )
    try:
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
