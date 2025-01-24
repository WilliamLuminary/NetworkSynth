# src/uni-processor_main.py
import logging
import warnings

import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config import Config, DataType, NameResolutionSet, Resolution, SetName
from graph import GraphAttrAgent, GraphGenerator, GraphPostProcessor
from handlers import DataAgent, MultifractalAnalyzer, Plotter, Saver
from utils import timer

Config.initialize()
logger = logging.getLogger(__name__)


@timer
def generate_and_process_graphs(data_agent: DataAgent):
    output_handler = None
    if not preview:
        output_handler = Saver(data_agent.name_res_set)
        output_handler.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)

    org_analyzer = MultifractalAnalyzer(data_agent.original_network)
    org_alpha_0, org_width = org_analyzer.multifractal_analysis()

    plot_agent = Plotter(data_agent.original_image, data_agent.original_network)
    graph = plot_agent.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
    )
    output_handler.save_file(graph, DataType.ORIGINAL_GRAPH)

    # Generate synthetic graphs
    max_attempts = Config.MAX_ATTEMPTS
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    error_threshold = Config.ERROR_TOLERANCE

    _errors = []

    synthetic_graph = None
    generator = GraphGenerator(data_agent.attributes)
    for _ in tqdm(range(num_syn_nw), desc="Generating Graphs"):

        error = float('inf')
        attempt = 0
        while error > error_threshold and attempt < max_attempts:
            attempt += 1

            _synthetic_graph = generator.generate_network()

            postprocessor = GraphPostProcessor(_synthetic_graph, data_agent.mapper,
                                               data_agent.attributes.avg_degree)
            postprocessor.trim_graph()
            postprocessor.assign_weights()
            synthetic_graph = postprocessor.synthetic_graph

            analyzer = MultifractalAnalyzer(synthetic_graph)
            alpha_0, width = analyzer.multifractal_analysis()
            error = euclidean([org_alpha_0, org_width], [alpha_0, width])

        if error > error_threshold:
            logger.warning(f"Failed to generate a valid synthetic graph after {max_attempts} attempts.")
            continue  # Skip this synthetic graph

        if num_syn_graph > 0:
            graph = plot_agent.plot_graph(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=synthetic_graph
            )
            output_handler.save_file(graph, DataType.SYNTHETIC_GRAPH)
            num_syn_graph -= 1

        if preview:
            return
        _errors.append(error)
        data_agent.add_synthetic_graph(synthetic_graph)

    output_handler.save_file(data_agent.synthetic_networks, DataType.SYNTHETIC_NETWORK,
                             file_name_prefix=f'error:{np.mean(_errors):.2f}')


save_plots = True
preview = False  # Set to True to only plot the original original_network
set_names = [SetName.D]
resolutions = [Resolution.X20K]

if __name__ == '__main__':
    if preview:
        warnings.warn("Only the original original_network will be shown WITHOUT SAVING")
    else:
        Saver.initialize()

    for set_name in set_names:
        for resolution in resolutions:
            name_res_set = NameResolutionSet(set_name, resolution)
            logger.info(f"Processing {name_res_set}")
            data_loader = DataAgent(name_res_set)
            data_loader.load_data()
            graph_attr = GraphAttrAgent(data_loader.original_network)
            data_loader.set_attributes(graph_attr)
            generate_and_process_graphs(data_loader)
