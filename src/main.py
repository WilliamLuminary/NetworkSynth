# src/main.py
import logging
import warnings

from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config.config import Config
from config.enums import DataType, Resolution, SetName
from config.name_resolution_set import NameResolutionSet
from handlers.data_agent import DataAgent
from graph.graph_attributes import GraphAttributes
from graph.graph_generator import GraphGenerator
from graph.graph_postprocessor import GraphPostProcessor
from handlers.output_handler import OutputHandler  # Import OutputHandler
from handlers.analyze_agent import AnalyzeAgent
from handlers.plot_agent import PlotAgent

Config.initialize()
logger = logging.getLogger(__name__)


def generate_and_process_graphs(
        data_agent: DataAgent,
        view_origin_figure_only: bool = False
):
    output_handler = None
    if view_origin_figure_only:
        warnings.warn("Only the original original_graph will be shown WITHOUT SAVING")
        # No output handler is needed in this case
    else:
        output_handler = OutputHandler(data_agent.name_res_set)
        output_handler.save_file(data_agent.original_image, DataType.ORIGINAL_IMAGE)

    org_analyzer = AnalyzeAgent(data_agent.original_graph)
    org_alpha_0, org_width = org_analyzer.multifractal_analysis()

    plot_agent = PlotAgent(data_agent)

    # Plot original original_graph
    graph = plot_agent.plot_graph(
        data_type=DataType.ORIGINAL_GRAPH,
        show=True,
        alpha=0.6
    )
    if view_origin_figure_only:
        return

    output_handler.save_file(graph, DataType.ORIGINAL_GRAPH)

    # Generate synthetic graphs
    max_attempts = Config.MAX_ATTEMPTS
    num_syn_nw = Config.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = Config.SYNTHETIC_GRAPH_NUMBER
    error_threshold = Config.ERROR_TOLERANCE

    synthetic_graph = None
    generator = GraphGenerator(data_agent)
    for _ in tqdm(range(num_syn_nw), desc="Generating Graphs", ncols=80):
        error = float('inf')
        attempt = 0

        while error > error_threshold and attempt < max_attempts:
            attempt += 1

            _synthetic_graph = generator.generate_network()

            postprocessor = GraphPostProcessor(_synthetic_graph, data_agent)
            postprocessor.remove_nodes_and_edges_by_degree_distribution()
            postprocessor.assign_weights()
            synthetic_graph = postprocessor.synthetic_graph

            analyzer = AnalyzeAgent(synthetic_graph)
            alpha_0, width = analyzer.multifractal_analysis()
            error = euclidean([org_alpha_0, org_width], [alpha_0, width])

        if attempt == max_attempts and error > error_threshold:
            print(f"Failed to generate a valid original_graph after {max_attempts} attempts.")
            continue  # Skip this iteration

        if num_syn_graph > 0:
            graph = plot_agent.plot_graph(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=synthetic_graph
            )
            output_handler.save_file(graph, DataType.SYNTHETIC_GRAPH)
            num_syn_graph -= 1

        data_agent.add_synthetic_graph(synthetic_graph)

    output_handler.save_file(data_agent.synthetic_graphs, DataType.SYNTHETIC_NETWORK)


save_plots = True
view_only = False  # Set to True to only plot the original original_graph
set_names = [SetName.D]
resolutions = [Resolution.X20K]

if __name__ == '__main__':
    OutputHandler.initialize()

    for set_name in set_names:
        for resolution in resolutions:
            name_res_set = NameResolutionSet(set_name, resolution)
            data_loader = DataAgent(name_res_set)
            data_loader.load_data()
            graph_attr = GraphAttributes(data_loader.original_graph)
            data_loader.set_attributes(graph_attr)
            logger.info(f"Processing {name_res_set}")
            generate_and_process_graphs(
                data_loader,
                save_plots=save_plots,
                view_origin_figure_only=view_only
            )
