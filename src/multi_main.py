# src/main.py
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config import Config, DataType, NameResolutionSet, Resolution, SetName
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver

Config.initialize()
logger = logging.getLogger(__name__)


def attempt_generate_graph(
        org_alpha_0: float,
        org_width: float,
        attributes,  # graph attributes (must be picklable)
        map_handler,  # from data_agent.mapper (must be picklable)
        avg_degree: float,
        error_threshold: float,
        max_attempts: int
):
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

        # 1) Generate the network
        _synthetic_graph = generator.generate_network()

        # 2) Post-process
        postprocessor = GraphPostProcessor(_synthetic_graph, map_handler, avg_degree)
        postprocessor.trim_graph()
        postprocessor.assign_weights()
        synthetic_graph = postprocessor.synthetic_graph

        # 3) Multifractal analysis
        analyzer = MultifractalAnalyzer(synthetic_graph)
        alpha_0, width = analyzer.multifractal_analysis()
        error = euclidean([org_alpha_0, org_width], [alpha_0, width])

    if error > error_threshold:
        # We failed to meet the threshold
        logger.warning(f"Failed after {max_attempts} attempts (error={error:.4f}).")
        return None, float('inf')

    # Success
    return synthetic_graph, error


def generate_and_process_graphs(data_agent: DataAgent):
    """
    Original function that now uses multiprocessing to generate synthetic graphs.
    """
    output_handler = None
    if not preview:
        output_handler = Saver(data_agent.name_res_set)
        output_handler.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)

    # Analyze the original graph
    org_analyzer = MultifractalAnalyzer(data_agent.original_graph)
    org_alpha_0, org_width = org_analyzer.multifractal_analysis()

    # Plot the original graph (in the main process)
    plot_agent = Plotter(data_agent.original_image, data_agent.original_graph)
    graph = plot_agent.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )
    if output_handler:
        output_handler.save_file(graph, DataType.ORIGINAL_GRAPH)

    # Generation parameters
    max_attempts = Config.MAX_ATTEMPTS
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    error_threshold = Config.ERROR_TOLERANCE

    # We'll track errors in a local queue
    _errors = []

    # We will gather results in the main process. So let's prepare a list of futures.
    futures = []

    # For multiprocessing, we need to ensure everything is picklable.
    # We'll pass only what is needed for each worker:
    attributes = data_agent.attributes  # GraphAttrAgent
    map_handler = data_agent.mapper  # Must be picklable
    avg_degree = attributes.avg_degree

    # Fire up a process pool
    with ProcessPoolExecutor() as executor:
        for _ in range(num_syn_nw):
            # Submit tasks to the pool
            future = executor.submit(
                attempt_generate_graph,
                org_alpha_0,
                org_width,
                attributes,  # from data_agent.attributes
                map_handler,  # from data_agent
                avg_degree,
                error_threshold,
                max_attempts
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
                # Reuse the same plot_agent or create a new one if needed
                graph = plot_agent.plot_graph(
                    data_type=DataType.SYNTHETIC_GRAPH,
                    graph=synthetic_graph
                )
                if output_handler:
                    output_handler.save_file(graph, DataType.SYNTHETIC_GRAPH)
                num_syn_graph -= 1

            if preview:
                return  # stop immediately if we're just previewing

            _errors.extend(error)
            data_agent.add_synthetic_graph(synthetic_graph)

    if output_handler:
        output_handler.save_file(
            list(data_agent.synthetic_graphs.queue),
            DataType.SYNTHETIC_NETWORK,
            file_name_prefix=f'error:{np.mean(_errors):.2f}'
        )


save_plots = True
preview = False  # Set to True to only plot the original graph
set_names = [SetName.D]
resolutions = [Resolution.X20K]

if __name__ == '__main__':
    if preview:
        warnings.warn("Only the original graph will be shown WITHOUT SAVING")
    else:
        Saver.initialize()

    for set_name in set_names:
        for resolution in resolutions:
            name_res_set = NameResolutionSet(set_name, resolution)
            logger.info(f"Processing {name_res_set}")

            data_loader = DataAgent(name_res_set)
            data_loader.load_data()

            # Create GraphAttrAgent from the original graph
            graph_attr = GraphAttrAgent(data_loader.original_graph)
            data_loader.set_attributes(graph_attr)

            # Call the (now) multiprocessing-enabled function
            generate_and_process_graphs(data_loader)
