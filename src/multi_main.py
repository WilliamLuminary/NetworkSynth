# src/main.py
import logging
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config import Config, DataType, NameResolutionSet, Resolution, SetName
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver
from utils import timer

Config.initialize()
logger = logging.getLogger(__name__)

error_threshold = Config.ERROR_TOLERANCE
max_attempts = Config.MAX_ATTEMPTS


def attempt_generate_graph(org_alpha_0_and_width: tuple[float, float], attributes, map_handler, avg_degree: float):
    """
    Worker function for generating a SINGLE synthetic graph that meets the
    (alpha_0, width) error threshold. Returns (synthetic_graph, error).
    If we fail after max_attempts, returns (None, float('inf')).
    """
    # Note: We create a new generator and postprocessor inside this function
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

        except Exception as e:
            logger.error(f"Exception occurred during graph generation: {e}. Skipping attempt {attempt}.")
            continue  # Skip this attempt and move to the next one

    if error > error_threshold:
        logger.warning(f"Failed after {max_attempts} attempts (error={error:.4f}).")
        return None, float('inf')

    return synthetic_graph, error


@timer
def run(name_res_set: NameResolutionSet) -> None:
    """
    Original function that now uses multiprocessing to generate synthetic graphs.
    """
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
    org_alpha_0, org_width = org_analyzer.multifractal_analysis()

    plotter = Plotter(data_agent.original_image, data_agent.original_graph)
    graph = plotter.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )
    if not preview:
        saver.save_file(graph, DataType.ORIGINAL_GRAPH)

    _error_dict = defaultdict(list)
    closed_nodes_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    closed_edges_factors = [round(0.1 + 0.1 * i, 1) for i in range(20)]
    for _nod_fac, _edg_fac in product(closed_nodes_factors, closed_edges_factors):
        Config.set_node_factor(_nod_fac)
        Config.set_edge_factor(_edg_fac)
        logger.info(Config())

        result = generate_synthetic(data_agent, plotter, saver, (org_alpha_0, org_width))
        if preview:
            return
        _error_dict[(_nod_fac, _edg_fac)].append(result)
        saver.save_file(_error_dict, DataType.DEFAULT_DATA, file_name_prefix='error_dict')


def generate_synthetic(data_agent: DataAgent, plotter: Plotter, saver, org_alpha_0_width: Tuple[float, float]) -> \
        Optional[float]:
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    _errors = []
    futures = []
    # For multiprocessing, we need to ensure everything is picklable.
    # We'll pass only what is needed for each worker:
    attributes = data_agent.attributes  # GraphAttrAgent
    map_handler = data_agent.mapper  # Must be picklable
    avg_degree = attributes.avg_degree
    with ProcessPoolExecutor() as executor:
        for _ in range(num_syn_nw):
            future = executor.submit(
                attempt_generate_graph,
                org_alpha_0_width,
                attributes,  # from data_agent.attributes
                map_handler,  # from data_agent
                avg_degree
            )
            futures.append(future)

        # Now, as each future completes, we handle the results
        for future in tqdm(as_completed(futures), total=num_syn_nw, desc="Generating Graphs"):
            synthetic_graph, error = future.result()

            # If synthetic_graph is None, generation failed or the threshold wasn't met
            if synthetic_graph is None:
                continue  # skip

            # Note: we do the plotting in the main process to avoid
            # potential issues with MPL in child processes.
            if num_syn_graph > 0:
                # Reuse the same plotter or create a new one if needed
                graph = plotter.plot_graph(
                    data_type=DataType.SYNTHETIC_GRAPH,
                    graph=synthetic_graph
                )
                if saver:
                    saver.save_file(graph, DataType.SYNTHETIC_GRAPH)
                num_syn_graph -= 1

            if preview:
                return None  # stop immediately if we're just previewing

            _errors.append(error)
            data_agent.add_synthetic_graph(synthetic_graph)
    _avg_err = np.mean(_errors)
    if saver:
        saver.save_file(
            data_agent.synthetic_graphs,
            DataType.SYNTHETIC_NETWORK,
            file_name_prefix=f'error:{_avg_err:.2f}'
        )
    return float(_avg_err)


save_plots = True
preview = False  # Set to True to only plot the original graph
set_names = [SetName.D]
resolutions = [Resolution.X20K]

if __name__ == '__main__':
    if preview:
        warnings.warn("Only the original graph will be shown WITHOUT SAVING")
    else:
        Saver.initialize()

    for _set_name, _resolution in product(set_names, resolutions):
        run(NameResolutionSet(_set_name, _resolution))
