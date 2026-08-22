import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# noinspection PyUnresolvedReferences
from dataclasses import replace
from itertools import product
from typing import Tuple

import wandb
from analysis import MultifractalAnalyzer
from configs import DatasetId, SynthParams
from configs.generate_mode import GenConfigTmp as GenConfig
from handlers import RunAgent, create_run_paths
from pipelines.generate import compute_average_error, generate_synthetic_network


class SweepRunConfig(GenConfig):

    SYNTHETIC_NETWORK_NUMBER = 100
    SYNTHETIC_GRAPH_NUMBER = 0
    DISABLE_SAVING = True
    DISABLE_SAVING_NOTE = "Sweeping Experiment"


SweepRunConfig.initialize()
run_paths = create_run_paths(SweepRunConfig)

EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"
EXPERIMENT_NAME = "mosaic-sample-sweep"
NODE_FACTORS = [round(0.4 + 0.1 * i, 1) for i in range(17)]  # 0.4 … 2.0
EDGE_FACTORS = [round(0.4 + 0.1 * i, 1) for i in range(17)]  # 0.4 … 2.0

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def generate_networks_multiprocess(
    data_agent: RunAgent, std_err_fea, trial_params
) -> Tuple[float, float]:
    num_network = SweepRunConfig.SYNTHETIC_NETWORK_NUMBER
    errors, futures = [], []
    from utils import spawn_context

    exit_event = spawn_context().Manager().Event()
    max_workers = SweepRunConfig.get_max_workers(num_network)
    logger.info(
        f"Spawning pool with {max_workers} workers for {num_network} networks …"
    )
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_context()
        ) as executor:
            futures = [
                executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    std_err_fea,
                    data_agent.attributes,
                    data_agent.mapper,
                    trial_params.for_worker(i),
                )
                for i in range(num_network)
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
    """Generate all networks for a single (nf, ef) pair.

    The pair lives in an immutable params object rather than being written into
    global config, so pairs cannot interfere with each other.
    """
    trial_params = replace(
        SynthParams.from_config(SweepRunConfig),
        closed_nodes_factor=nf,
        closed_edges_factor=ef,
    )
    logger.info(f"nf={nf}, ef={ef}")
    return generate_networks_multiprocess(data_agent, std_err_fea, trial_params)


def run_for_dataset(dataset_id: DatasetId) -> None:
    logger.info(f"Processing dataset: {dataset_id}")
    data_agent = RunAgent(SweepRunConfig, run_paths=run_paths, dataset_id=dataset_id)
    data_agent.prepare_data()
    std_err_fea = MultifractalAnalyzer(
        data_agent.get_original_network(),
        SweepRunConfig.MEASURE_WEIGHTED,
        SweepRunConfig.FULL_Q_BAND,
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
        SweepRunConfig.SYNTHETIC_NETWORK_NUMBER != 0
    ), "Sweeping experiments require synthetic networks."

    max_workers = SweepRunConfig.get_max_workers(
        SweepRunConfig.SYNTHETIC_NETWORK_NUMBER
    )
    logger.info(
        f"Using {max_workers} worker(s) for {SweepRunConfig.SYNTHETIC_NETWORK_NUMBER} networks per param pair"
    )

    wandb.login()
    wandb.init(
        project=EXPERIMENT_PROJECT_NAME,
        name=EXPERIMENT_NAME,
        dir=SweepRunConfig.PROJECT_ROOT,
    )
    try:
        for dataset_id in SweepRunConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
