# src/pipelines/sweep.py
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import wandb
from analysis import MultifractalAnalyzer

# noinspection PyUnresolvedReferences
from configs import BaseConfig, DatasetId
from handlers import RunAgent
from pipelines.generate import compute_average_error, generate_synthetic_network

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
NODE_FACTORS = [round(0.3 + 0.1 * i, 1) for i in range(18)]
EDGE_FACTORS = [round(0.3 + 0.1 * i, 1) for i in range(18)]

logger = logging.getLogger(__name__)


def _generate_with_factors(exit_event, std_err_fea, attributes, mapper, nf, ef):
    """Spawn-safe wrapper: explicitly set factors before generation."""
    BaseConfig.CLOSED_NODES_FACTOR = nf
    BaseConfig.CLOSED_EDGES_FACTOR = ef
    return generate_synthetic_network(exit_event, std_err_fea, attributes, mapper)


def generate_networks(data_agent, std_err_fea, nf, ef):
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
                    std_err_fea,
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


def run_for_dataset(dataset_id: DatasetId) -> None:
    """Create a wandb sweep for a single dataset and run all trials."""
    logger.info(f"Processing dataset: {dataset_id}")
    data_agent = RunAgent(dataset_id=dataset_id)
    data_agent.prepare_data()
    std_err_fea = MultifractalAnalyzer(
        data_agent.get_original_network()
    ).analyze_error_features()

    sweep_config = {
        "name": f"sweep-{dataset_id}",
        "method": "grid",
        "metric": {"name": "error", "goal": "minimize"},
        "parameters": {
            "node_factor": {"values": NODE_FACTORS},
            "edge_factor": {"values": EDGE_FACTORS},
        },
    }

    def trial():
        with wandb.init():
            nf = wandb.config.node_factor
            ef = wandb.config.edge_factor
            logger.info(f"Trial: nf={nf}, ef={ef}")

            error, success_rate = generate_networks(data_agent, std_err_fea, nf, ef)
            if error is None:
                error = float("inf")

            wandb.log({"error": error, "success_rate": success_rate})

    sweep_id = wandb.sweep(sweep_config, project=EXPERIMENT_PROJECT_NAME)
    wandb.agent(sweep_id, function=trial)


def main(config=None):
    _init_config(config)

    assert (
        BaseConfig.SYNTHETIC_NETWORK_NUMBER != 0
    ), "Sweeping experiments require synthetic networks."

    wandb.login()
    try:
        for dataset_id in BaseConfig.get_datasets():
            run_for_dataset(dataset_id)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == "__main__":
    main()
