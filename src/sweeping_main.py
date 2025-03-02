# src/sweeping_main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional

import cv2

import wandb
from analysis import MultifractalAnalyzer
from config import BaseConfig, DataType, Resolution, SetName
# noinspection PyUnresolvedReferences
from handlers import DataAgent
from handlers.data_agent import plot_network
from config import GenConfig, GenConfig1, GenConfig2
from main import compute_average_error, generate_synthetic_network

GenConfig2.initialize()
BaseConfig.disable_saving("Sweeping Experiment")
EXPERIMENT_PROJECT_NAME = "hyperparam-tuning"
EXPERIMENT_NAME = "adjusting-node-and-edge-factors"
NODE_FACTORS = [round(0.1 + 0.1 * i, 1) for i in range(20)]
EDGE_FACTORS = [round(0.1 + 0.1 * i, 1) for i in range(20)]

logger = logging.getLogger(__name__)
SIGINT_INFO = "SIGINT received. Terminating child process..."


def generate_networks_multiprocess(data_agent: DataAgent, std_err_fea) -> Optional[float]:
    num_network, num_figures = BaseConfig.SYNTHETIC_NETWORK_NUMBER, BaseConfig.SYNTHETIC_GRAPH_NUMBER
    errors, futures = [], []
    from multiprocessing import Manager
    exit_event = Manager().Event()
    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(generate_synthetic_network, exit_event, std_err_fea,
                                       data_agent.attributes, data_agent.mapper) for _ in range(num_network)]
            next_log = 0
            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error = future.result()
                if not synthetic_graph:
                    continue

                if (progress := round(idx / num_network * 100, 2)) >= (next_log := next_log + 10):
                    logger.info(f"({progress}%) Synthetic graph generated.")

                if (num_figures := num_figures - 1) >= 0:
                    synthetic_graph_image = plot_network(data_type=DataType.SYNTHETIC_GRAPH, graph=synthetic_graph)
                    _, img_encoded = cv2.imencode('.png', synthetic_graph_image)
                    wandb.log({"synthetic_graph": wandb.Image(img_encoded.tobytes())})
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
        executor.shutdown(wait=True, cancel_futures=True)

    return compute_average_error(errors)


def run_with_params(data_agent, ef, nf, std_err_fea):
    BaseConfig.set_node_factor(nf)
    BaseConfig.set_edge_factor(ef)
    logger.info(BaseConfig())
    error = generate_networks_multiprocess(data_agent, std_err_fea)
    return error


def run_for_each_set_resolution(set_name: SetName, resolution: Resolution) -> None:
    data_agent = DataAgent(set_name=set_name, resolution=resolution)
    data_agent.prepare_data()
    std_err_fea = MultifractalAnalyzer(data_agent.get_original_network()).analyze_error_features()

    table = wandb.Table(columns=["node_factor", "edge_factor", "error"])
    for nf, ef in product(NODE_FACTORS, EDGE_FACTORS):
        error = run_with_params(data_agent, ef, nf, std_err_fea)
        if error is not None:
            table.add_data(nf, ef, error)
        else:
            logger.warning(f"Error is None for node_factor {nf}, edge_factor {ef}")

    wandb.log({"my_heatmap": wandb.plot_table("heatmap", table,
                                              {"x": "node_factor", "y": "edge_factor", "value": "error"}
                                              )})


def main():
    assert BaseConfig.SYNTHETIC_NETWORK_NUMBER != 0, "Sweeping experiments require synthetic networks."

    wandb.login()
    wandb.init(project=EXPERIMENT_PROJECT_NAME, name=EXPERIMENT_NAME, dir=BaseConfig.PROJECT_ROOT)
    try:
        for set_name_, resolution_ in product(BaseConfig.SETS, BaseConfig.RESOLUTIONS):
            run_for_each_set_resolution(set_name_, resolution_)
    except KeyboardInterrupt:
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")


if __name__ == '__main__':
    main()
