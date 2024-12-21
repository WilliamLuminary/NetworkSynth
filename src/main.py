# src/main.py
import logging
import os
import warnings

import cv2
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config.base import BaseConfig
from config.enums import SetName, Resolution
from config.name_resolution_set import NameResolutionSet
from data.graph_data_agent import GraphDataAgent
from utils.graph_analysis import GraphAnalyzer
from graph.graph_generator import GraphGenerator
from graph.graph_postprocessor import GraphPostProcessor
from graph.graph_attributes import GraphAttributes
from utils.debug_utils import debugging
from config.output_handler import OutputHandler  # Import OutputHandler
from utils.plotting_utils import plot_graph

BaseConfig.initialize()
logger = logging.getLogger(__name__)


@debugging
def generate_and_process_graphs(
        data_loader: GraphDataAgent,
        save_plots=True,
        view_origin_figure_only: bool = False,
        error_threshold=1.5):
    if view_origin_figure_only:
        warnings.warn("Only the original original_graph will be shown WITHOUT SAVING")
        # No output handler is needed in this case
    else:
        output_handler = OutputHandler(data_loader.name_res_set)
        logging.info(f"Set up the output directory as {output_handler.base_output_dir}")

    org_analyzer = GraphAnalyzer(data_loader.original_graph)
    org_analyzer.multi_analysis()

    synthetic_graph_frame = [[0, BaseConfig.DEFAULT_FRAME_RANGE], [0, BaseConfig.DEFAULT_FRAME_RANGE]]
    original_graph_output_path = os.path.join(BaseConfig.ORIGINAL_GRAPH_PATH, f"Set {set_name} Res {resolution}")
    synthetic_graph_output_path = os.path.join(BaseConfig.SYNTHETIC_GRAPH_PATH, f"Set {set_name} Res {resolution}")

    # Plot original original_graph
    if view_origin_figure_only or save_plots:
        plot_graph(
            graph=data_loader.original_graph,
            set_name=set_name,
            resolution=resolution,
            frame=synthetic_graph_frame,
            save=save_plots and not view_origin_figure_only,
            output_path=original_graph_output_path,
            background=True,
            image=data_loader.original_image,
            alpha=0.6,
            title='Original Graph'
        )
        cv2.imwrite(os.path.join(original_graph_output_path, f'Original Image - Set {set_name} Res {resolution}.png'),
                    data_loader.original_image)
        if view_origin_figure_only:
            return

    # Generate synthetic graphs
    synthetic_graphs = []
    errors = []

    max_attempts = BaseConfig.MAX_ATTEMPTS
    num_syn_nw = BaseConfig.SYNTHETIC_NETWORK_NUMBER
    num_syn_graph = BaseConfig.SYNTHETIC_GRAPH_NUMBER
    for i in tqdm(range(num_syn_nw), desc="Generating Graphs", ncols=80):
        error = float('inf')
        attempt = 0

        while error > error_threshold and attempt < max_attempts:
            attempt += 1

            generator = GraphGenerator(data_loader)
            synthetic_graph, synthetic_graph_frame = generator.generate_network()

            postprocessor = GraphPostProcessor(synthetic_graph, data_loader)
            postprocessor.adjust_degree_distribution()
            postprocessor.assign_weights(map_length_to_weight, length_bins, weight_baskets, y)
            processed_graph = postprocessor.synthetic_graph

            # Calculate multifractal properties
            graph_analyzer = GraphAnalyzer(processed_graph)
            tau_list = graph_analyzer.calculate_multifractal_spectrum(Q)
            alpha_0, width, al_list, fal_list = graph_analyzer.n_spectrum(tau_list, Q)

            # Calculate error
            error = euclidean([org_analyzer.alpha_0, org_analyzer.width], [alpha_0, width])

        if attempt == max_attempts and error > error_threshold:
            print(f"Failed to generate a valid original_graph after {max_attempts} attempts.")
            continue  # Skip this iteration

        synthetic_graphs.append(processed_graph)
        errors.append(error)

        # Plot and save a sample of graphs
        if save_plots and i < num_syn_graph:
            plot_graph(
                graph=processed_graph,
                set_name=set_name,
                resolution=resolution,
                frame=synthetic_graph_frame,
                save=True,
                output_path=synthetic_graph_output_path,
                title=f"Synthetic Graph {i + 1}"
            )
            print(f"Plotted and saved synthetic original_graph {i + 1} with error {error:.4f}")
        else:
            print(f"Generated synthetic original_graph {i + 1} with error {error:.4f}")

    # Save synthetic graphs
    save_dir = os.path.join(config.OUTPUT_DIR, f"Set {set_name} Res {resolution}")
    output_handler.ensure_directory(save_dir)

    save_path = os.path.join(save_dir, 'synthetic_graphs.pkl')
    output_handler.save_pickle(synthetic_graphs, save_path)

    # Save errors
    errors_path = os.path.join(save_dir, 'synthetic_graph_errors.pkl')
    output_handler.save_pickle(errors, errors_path)

    print(f"Synthetic graphs and errors saved to {save_dir}")


save_plots = True
view_only = False  # Set to True to only plot the original original_graph
set_names = [SetName.D]
resolutions = [Resolution.X20K]

if __name__ == '__main__':
    for set_name in set_names:
        for resolution in resolutions:
            name_res_set = NameResolutionSet(set_name, resolution)
            data_loader = GraphDataAgent(name_res_set)
            data_loader.load_data()
            attributes = GraphAttributes(data_loader.original_graph)
            data_loader.set_attributes(attributes)
            logger.info(f"Processing {name_res_set}")
            generate_and_process_graphs(
                data_loader,
                save_plots=save_plots,
                view_origin_figure_only=view_only
            )
