# src/main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

import wandb
from config import Config, DataType, Resolution, SetName
# noinspection PyUnresolvedReferences
from config import Config1, Config2, ConfigSample
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver, Summary

Config2.initialize()
# Config2.initialize()
preview = False
exp = False  # Set to True to run the hyperparameter tuning experiment
# Config.disable_saving("Debugging")

logger = logging.getLogger(__name__)


def generate_synthetic_network(exit_event, org_alpha_0_and_width: tuple[float, float], attributes, map_handler,
                               avg_degree: float):
    generator = GraphGenerator(attributes)

    _error = float('inf')
    _attempt = 0
    _synthetic_graph = None
    _error_threshold = Config.ERROR_TOLERANCE
    _max_attempts = Config.MAX_ATTEMPTS
    while (_error > _error_threshold
           and _attempt < _max_attempts
           and not exit_event.is_set()):
        _attempt += 1

        try:
            _synthetic_graph = generator.generate_network()

            _postprocessor = GraphPostProcessor(_synthetic_graph, map_handler, avg_degree)
            _postprocessor.trim_graph()
            _postprocessor.assign_weights()
            _synthetic_graph = _postprocessor.synthetic_graph

            _analyzer = MultifractalAnalyzer(_synthetic_graph)
            alpha_0_and_width = _analyzer.multifractal_analysis()

            _error = euclidean(org_alpha_0_and_width, alpha_0_and_width)

        except Exception as exc:
            logger.error(f"Exception occurred during graph generation: {exc}. Skipping attempt {_attempt}.")
            continue

    if exit_event.is_set():
        logger.debug("Child process exiting due to exit signal.")
        return None, float('inf')

    if _error > _error_threshold:
        logger.warning(f"Failed after {_max_attempts} attempts (error={_error:.4f}).")
        return None, float('inf')

    return _synthetic_graph, _error


def generate_with_multiprocessing(data_agent: DataAgent, plotter: Plotter, saver, org_alpha_0_width: Tuple[float, float]) -> \
        Optional[float]:
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    _errors = []
    futures = []
    attributes = data_agent.attributes
    map_handler = data_agent.mapper
    avg_degree = attributes.average_degree

    from multiprocessing import Manager
    manager = Manager()
    exit_event = manager.Event()

    try:
        with ProcessPoolExecutor() as executor:
            for _ in range(num_syn_nw):
                future = executor.submit(
                    generate_synthetic_network,
                    exit_event,
                    org_alpha_0_width,
                    attributes,
                    map_handler,
                    avg_degree
                )
                futures.append(future)

            for idx, future in enumerate(as_completed(futures), start=1):
                synthetic_graph, _error = future.result()
                logger.info(f"[{idx}/{num_syn_nw}] Synthetic graph generation completed.")

                if synthetic_graph is None:
                    continue

                # Note: we do the plotting in the main process to avoid potential issues with MPL in child processes.
                if num_syn_graph > 0:
                    graph = plotter.plot_graph(
                        data_type=DataType.SYNTHETIC_GRAPH,
                        graph=synthetic_graph
                    )
                    if saver:
                        saver.save_file(graph, DataType.SYNTHETIC_GRAPH)
                    num_syn_graph -= 1

                if preview:
                    print(
                        f'Node Factor {Config.CLOSED_NODES_FACTOR}. '
                        f'Edge Factor {Config.CLOSED_EDGES_FACTOR}. '
                        f'Error: {_error:.4f}')
                    return None

                _errors.append(_error)
                data_agent.add_synthetic_graph(synthetic_graph)

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

    if not _errors:
        return float('inf')

    _non_inf_errors = [__e for __e in _errors if not np.isinf(__e)]
    if not _non_inf_errors:
        return float('inf')

    _mean_e = np.mean(_non_inf_errors)
    _std_e = np.std(_non_inf_errors)
    _threshold = 2.0
    _non_outlier_errors = [__e for __e in _non_inf_errors if abs(__e - _mean_e) <= _threshold * _std_e]
    if not _non_outlier_errors:
        return float('inf')

    _avg_err = round(np.mean(_non_outlier_errors), 3)

    if saver:
        saver.save_file(
            data_agent.synthetic_networks,
            DataType.SYNTHETIC_NETWORK,
            file_name_prefix=f'nw_no_{num_syn_nw}_err_{_avg_err:.3f}'
        )
    return _avg_err


def run(set_name: SetName, resolution: Resolution) -> Optional[float]:
    logger.info(Config())
    data_agent = DataAgent(set_name, resolution)
    data_agent.load_data()
    attr = GraphAttrAgent()
    attr.analyze(data_agent.original_network)
    data_agent.set_attributes(attr)
    saver = None
    if not preview:
        saver = Saver(data_agent.set_name, data_agent.resolution)
        saver.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)
        saver.save_file(data_agent.original_network, DataType.ORIGINAL_NETWORK)
        saver.save_file(data_agent.attributes, DataType.ORIGINAL_PROPERTY)

    plotter = Plotter(data_agent.original_image, data_agent.original_network)
    graph = plotter.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )

    if Config.SYNTHETIC_NETWORK_NUMBER != 0:
        org_analyzer = MultifractalAnalyzer(data_agent.original_network)
        mult_ans_res = org_analyzer.multifractal_analysis()
    else:
        mult_ans_res = [0, 0]

    if not preview:
        saver.save_file(graph, DataType.ORIGINAL_GRAPH)

    if exp:
        exp_hyper_tuning(data_agent, plotter, saver, mult_ans_res)
        return None
    else:
        _error = generate_with_multiprocessing(data_agent, plotter, saver, mult_ans_res)
        return _error


def exp_hyper_tuning(data_agent, plotter, saver, mult_ans_res):
    logger.info("[EXP] This is an experiment")
    closed_nodes_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    closed_edges_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    Config.disable_saving("Hyperparameter tuning")
    table = wandb.Table(columns=["node_factor", "edge_factor", "error"])
    for __nf, __ef in product(closed_nodes_factors, closed_edges_factors):
        Config.set_node_factor(__nf)
        Config.set_edge_factor(__ef)
        logger.info(Config())

        _error = generate_with_multiprocessing(data_agent, plotter, saver, mult_ans_res)
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

        summary = Summary()

        for __set_name, __resolution in product(set_names, resolutions):
            error = run(__set_name, __resolution)
            if error is not None:
                summary.add_result(__set_name.value, __resolution.value, error)

        summary.summarize()

    except Exception as e:
        logger.exception(e)
        raise

## Node before 1.0, edge before 1.o
