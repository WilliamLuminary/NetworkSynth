# src/main.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import wandb
from config import Config, DataType, NameResolutionSet, Resolution, SetName
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver
from utils import timer

Config.initialize()
logger = logging.getLogger(__name__)

error_threshold = Config.ERROR_TOLERANCE
max_attempts = Config.MAX_ATTEMPTS


def multi_generate_synthetic(org_alpha_0_and_width: tuple[float, float], attributes, map_handler, avg_degree: float):
    generator = GraphGenerator(attributes)

    error = float('inf')
    attempt = 0
    synthetic_graph = None

    while error > error_threshold and attempt < max_attempts:
        attempt += 1

        try:
            _synthetic_graph = generator.generate_network()

            postprocessor = GraphPostProcessor(_synthetic_graph, map_handler, avg_degree)
            postprocessor.trim_graph()
            postprocessor.assign_weights()
            synthetic_graph = postprocessor.synthetic_graph

            analyzer = MultifractalAnalyzer(synthetic_graph)
            alpha_0, width = analyzer.multifractal_analysis()

            error = euclidean(org_alpha_0_and_width, [alpha_0, width])

        except Exception as exc:
            logger.error(f"Exception occurred during graph generation: {exc}. Skipping attempt {attempt}.")
            continue

    if error > error_threshold:
        logger.warning(f"Failed after {max_attempts} attempts (error={error:.4f}).")
        return None, float('inf')

    return synthetic_graph, error


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

        for future in tqdm(as_completed(futures), total=num_syn_nw, desc="Generating Graphs"):
            synthetic_graph, error = future.result()

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
                    f'Error: {error:.4f}')
                return None

            _errors.append(error)
            data_agent.add_synthetic_graph(synthetic_graph)
    if _errors:
        _avg_err = np.mean(_errors)
    else:
        return float('inf')

    if saver:
        saver.save_file(
            data_agent.synthetic_graphs,
            DataType.SYNTHETIC_NETWORK,
            file_name_prefix=f'error:{_avg_err:.2f}'
        )
    return float(_avg_err)


@timer
def run(name_res_set: NameResolutionSet) -> None:
    logger.info(f"Processing {name_res_set}")
    data_agent = DataAgent(name_res_set)
    data_agent.load_data()
    graph_attr = GraphAttrAgent(data_agent.original_graph)
    data_agent.set_attributes(graph_attr)
    saver = None
    if not preview:
        saver = Saver(data_agent.name_res_set)
        saver.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)

    org_analyzer = MultifractalAnalyzer(data_agent.original_graph)
    mult_ans_res = org_analyzer.multifractal_analysis()

    plotter = Plotter(data_agent.original_image, data_agent.original_graph)
    graph = plotter.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )
    if not preview:
        saver.save_file(graph, DataType.ORIGINAL_GRAPH)

    generate_synthetic(data_agent, plotter, saver, mult_ans_res)
    # exp_hyper_tuning(data_agent, plotter, saver, mult_ans_res)


def exp_hyper_tuning(data_agent, plotter, saver, mult_ans_res):
    closed_nodes_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    closed_edges_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    logger.info("[EXP] This is an experiment")
    Config.disable_saving("Hyperparameter tuning")
    for _nod_fac, _edg_fac in product(closed_nodes_factors, closed_edges_factors):
        Config.disable_saving()
        Config.set_node_factor(_nod_fac)
        Config.set_edge_factor(_edg_fac)
        logger.info(Config())

        _error = generate_synthetic(data_agent, plotter, saver, mult_ans_res)
        if preview:
            logger.info("Preview mode enabled, skipping further iterations.")
            return
        if _error is not None:
            wandb.log({"node_factor": _nod_fac, "edge_factor": _edg_fac, "error": _error}, step=None)
        else:
            logger.warning(f"Error is None for node_factor {_nod_fac}, edge_factor {_edg_fac}")


save_plots = False
preview = False
set_names = [SetName.A]
resolutions = [Resolution.X10K]

if __name__ == '__main__':
    if not preview:
        wandb.login()
        wandb.init(project="graph-hyperparam-tuning", name="hyperparam_exp")
    try:
        if preview:
            warn_mesg = ("Preview Mode: Only the original image and graph and 1 synthetic graph "
                         "will be shown WITHOUT SAVING")
            logger.warning(warn_mesg)
        else:
            Saver.initialize()

        for _set_name, _resolution in product(set_names, resolutions):
            run(NameResolutionSet(_set_name, _resolution))
    except Exception as e:
        logger.exception(e)
        raise

## Node before 1.0, edge before 1.o
