# src/main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

import wandb
from config import Config, DataType, NameResolutionSet, Resolution, SetName
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver, Summary

Config.initialize()
logger = logging.getLogger(__name__)

error_threshold = Config.ERROR_TOLERANCE
max_attempts = Config.MAX_ATTEMPTS


def multi_generate_synthetic(org_alpha_0_and_width: tuple[float, float], attributes, map_handler, avg_degree: float):
    generator = GraphGenerator(attributes)

    __error = float('inf')
    attempt = 0
    synthetic_graph = None

    while __error > error_threshold and attempt < max_attempts:
        attempt += 1

        try:
            _synthetic_graph = generator.generate_network()

            postprocessor = GraphPostProcessor(_synthetic_graph, map_handler, avg_degree)
            postprocessor.trim_graph()
            postprocessor.assign_weights()
            synthetic_graph = postprocessor.synthetic_graph

            analyzer = MultifractalAnalyzer(synthetic_graph)
            alpha_0_and_width = analyzer.multifractal_analysis()

            __error = euclidean(org_alpha_0_and_width, alpha_0_and_width)

        except Exception as exc:
            logger.error(f"Exception occurred during graph generation: {exc}. Skipping attempt {attempt}.")
            continue

    if __error > error_threshold:
        logger.warning(f"Failed after {max_attempts} attempts (error={__error:.4f}).")
        return None, float('inf')

    return synthetic_graph, __error


def generate_synthetic(data_agent: DataAgent, plotter: Plotter, saver, org_alpha_0_width: Tuple[float, float]) -> \
        Optional[float]:
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    _errors = []
    futures = []
    attributes = data_agent.attributes
    map_handler = data_agent.mapper
    avg_degree = attributes.avg_degree
    with ProcessPoolExecutor() as executor:
        for _ in range(num_syn_nw):
            future = executor.submit(
                multi_generate_synthetic,
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

    if not _errors:
        return float('inf')

    __non_inf_errors = [__e for __e in _errors if not np.isinf(__e)]
    if not __non_inf_errors:
        return float('inf')

    __mean_e = np.mean(__non_inf_errors)
    __std_e = np.std(__non_inf_errors)
    __threshold = 2.0
    __non_outlier_errors = [__e for __e in __non_inf_errors if abs(__e - __mean_e) <= __threshold * __std_e]
    if not __non_outlier_errors:
        return float('inf')

    _avg_err = float(np.mean(__non_outlier_errors))

    if saver:
        saver.save_file(
            data_agent.synthetic_graphs,
            DataType.SYNTHETIC_NETWORK,
            file_name_prefix=f'error:{_avg_err:.2f}'
        )
    return _avg_err


# @timer
def run(name_res_set: NameResolutionSet) -> Optional[float]:
    logger.info(f"Processing {name_res_set}")
    data_agent = DataAgent(name_res_set)
    data_agent.load_data()
    graph_attr = GraphAttrAgent(data_agent.original_graph)
    data_agent.set_attributes(graph_attr)
    saver = None
    if not preview:
        saver = Saver(data_agent.name_res_set)
        saver.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)

    plotter = Plotter(data_agent.original_image, data_agent.original_graph)
    graph = plotter.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )

    if Config.SYNTHETIC_NETWORK_NUMBER != 0:
        org_analyzer = MultifractalAnalyzer(data_agent.original_graph)
        mult_ans_res = org_analyzer.multifractal_analysis()
    else:
        mult_ans_res = [0, 0]

    if not preview:
        saver.save_file(graph, DataType.ORIGINAL_GRAPH)

    if exp:
        exp_hyper_tuning(data_agent, plotter, saver, mult_ans_res)
        return None
    else:
        _error = generate_synthetic(data_agent, plotter, saver, mult_ans_res)
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

        _error = generate_synthetic(data_agent, plotter, saver, mult_ans_res)
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


preview = False
exp = False  # Set to True to run the hyperparameter tuning experiment
set_names = [SetName.S4, SetName.S8, SetName.S11, SetName.S14, SetName.S17, SetName.S20, SetName.S23, SetName.S26,
             SetName.S29, SetName.S32]
# set_names = [SetName.A]
resolutions = [Resolution.NA]
# resolutions = [Resolution.X20K]
# Config.disable_saving("Debugging")

if __name__ == '__main__':
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
            error = run(NameResolutionSet(__set_name, __resolution))
            if error is not None:
                summary.add_result(__set_name.value, __resolution.value, error)
        summary.summarize()

    except Exception as e:
        logger.exception(e)
        raise

## Node before 1.0, edge before 1.o
