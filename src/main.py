# src/main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional

import numpy as np

import wandb
from analysis import MultifractalAnalyzer
from config import Config, DataType, Resolution, SetName
# noinspection PyUnresolvedReferences
from config import Config1, Config2, ConfigSample
from graph import GraphGenerator
from utils.utils import trim_graph
from handlers import DataAgent, Saver, Summarizer

Config2.initialize()

preview = False
exp = False  # Set to True to run the hyperparameter tuning experiment
if exp:
    Config.disable_saving("Experiment")
if preview:
    Config.disable_saving("Preview")

logger = logging.getLogger(__name__)



def generate_synthetic_network(exit_event, std_err_fea, attributes, mapper):
    generator = GraphGenerator(attributes)

    error_ = float('inf')
    attempt = 0
    synthetic_graph = None
    error_threshold = Config.ERROR_TOLERANCE
    max_attempts = Config.MAX_ATTEMPTS
    while (error_ > error_threshold
           and attempt < max_attempts
           and not exit_event.is_set()):
        attempt += 1

        try:
            synthetic_graph = generator.generate_network()

            if exit_event.is_set():
                logger.info(SIGINT_INFO)
                return None, float('inf')

            synthetic_graph = trim_graph(synthetic_graph, attributes.average_degree)
            mapper.assign_weights(synthetic_graph)

            if exit_event.is_set():
                logger.info(SIGINT_INFO)
                return None, float('inf')

            analyzer = MultifractalAnalyzer(synthetic_graph)
            err_fea = analyzer.analyze_error_values()

            error_ = analyzer.analyze_error(err_fea, std_err_fea)

        except Exception as exc:
            logger.error(f"Exception occurred during graph generation: {exc}. Skipping attempt {attempt}.")
            continue

    if exit_event.is_set():
        logger.debug("Child process exiting due to exit signal.")
        return None, float('inf')

    if error_ > error_threshold:
        logger.warning(f"Failed after {max_attempts} attempts (error={error_:.4f}).")
        return None, float('inf')

    return synthetic_graph, error_


def generate_with_multiprocessing(data_agent: DataAgent, std_err_fea) -> \
        Optional[float]:
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    errors = []
    futures = []

    from multiprocessing import Manager
    manager = Manager()
    exit_event = manager.Event()

    try:
        with ProcessPoolExecutor() as executor:
            for _ in range(num_syn_nw):
                future = executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    std_err_fea,
                    data_agent.attributes,
                    data_agent.mapper
                )
                futures.append(future)

            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, error_ = future.result()
                logger.info(f"[{idx}/{num_syn_nw}] Synthetic graph generation completed.")

                if synthetic_graph is None:
                    continue

                # Note: we do the plotting in the main process to avoid potential issues with MPL in child processes.
                if num_syn_graph > 0:
                    data_agent.save(data_type=DataType.SYNTHETIC_GRAPH, arg=synthetic_graph)
                    num_syn_graph -= 1
                else:
                    data_agent.add_synthetic_graph(synthetic_graph)

                errors.append(error_)

                if preview:
                    print(
                        f'Node Factor {Config.CLOSED_NODES_FACTOR}. '
                        f'Edge Factor {Config.CLOSED_EDGES_FACTOR}. '
                        f'Error: {error_:.4f}')
                    return None

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Terminating processes...")
        exit_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False)
        raise

    except Exception:
        exit_event.set()
        executor.shutdown(wait=False)
        raise

    if not errors:
        return float('inf')

    _non_inf_errors = [__e for __e in errors if not np.isinf(__e)]
    if not _non_inf_errors:
        return float('inf')

    _mean_e = np.mean(_non_inf_errors)
    _std_e = np.std(_non_inf_errors)
    _threshold = 2.0
    _non_outlier_errors = [__e for __e in _non_inf_errors if abs(__e - _mean_e) <= _threshold * _std_e]
    if not _non_outlier_errors:
        return float('inf')

    _avg_err = round(np.mean(_non_outlier_errors), 3)

    data_agent.save(data_type=DataType.SYNTHETIC_NETWORK, file_name_prefix=f'nw_no_{num_syn_nw}_err_{_avg_err:.3f}')

    return _avg_err


def run(set_name: SetName, resolution: Resolution) -> Optional[float]:
    logger.info(Config())
    data_agent = DataAgent(set_name, resolution)
    data_agent.load_data()
    attr = GraphAttrAgent()
    attr.analyze(data_agent.original_network)
    data_agent.set_attributes(attr)
    data_agent.save(DataType.ORIGINAL_IMAGE)
    data_agent.save(DataType.ORIGINAL_NETWORK)
    data_agent.save(DataType.ORIGINAL_PROPERTY)

    if Config.SYNTHETIC_NETWORK_NUMBER != 0:
        org_analyzer = MultifractalAnalyzer(data_agent.original_network)
        std_err_fea = org_analyzer.analyze_error_values()
    else:
        std_err_fea = [0, 0]

    data_agent.save(DataType.ORIGINAL_GRAPH)

    if exp:
        exp_hyper_tuning(data_agent, std_err_fea)
        return None
    else:
        _error = generate_with_multiprocessing(data_agent, std_err_fea)
        return _error


def exp_hyper_tuning(data_agent, std_err_fea):
    logger.info("[EXP] This is an experiment")
    closed_nodes_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    closed_edges_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    Config.disable_saving("Hyperparameter tuning")
    table = wandb.Table(columns=["node_factor", "edge_factor", "error"])
    for __nf, __ef in product(closed_nodes_factors, closed_edges_factors):
        Config.set_node_factor(__nf)
        Config.set_edge_factor(__ef)
        logger.info(Config())

        _error = generate_with_multiprocessing(data_agent, std_err_fea)
        if preview:
            logger.info("Preview mode enabled, skipping further iterations.")
            return
        if _error is not None:
            table.add_data(__nf, __ef, _error)
        else:
            logger.warning(f"Error is None for node_factor {__nf}, edge_factor {__ef}")

    heatmap_plot = wandb.plot_table(
        vega_spec_name="heatmap",
        data_table=table,
        fields={"x": "node_factor", "y": "edge_factor", "value": "error"}
    )
    wandb.log({"my_heatmap": heatmap_plot})


if __name__ == '__main__':
    set_names = Config.SETS
    resolutions = Config.RESOLUTIONS
    if not preview and exp:
        wandb.login()
        wandb.init(project="graph-hyperparam-tuning", name="hyperparam_exp")
    try:
        if preview:
            warn_mesg = (
                "Preview Mode: Only the original image and graph "
                "and 1 synthetic graph will be shown WITHOUT SAVING"
            )
            logger.warning(warn_mesg)
        else:
            Saver.initialize()

        summary = Summarizer()

        for __set_name, __resolution in product(set_names, resolutions):
            error = run(__set_name, __resolution)
            if error is not None:
                summary.add_result(__set_name.value, __resolution.value, error)

        summary.summarize()

    except Exception as e:
        logger.exception(e)
        raise

## Node before 1.0, edge before 1.o
