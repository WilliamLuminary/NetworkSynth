# src/main.py

import os
import warnings

import cv2
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config.config import Config
from data.data_loader import DataLoader
from graph.graph_analyzer import GraphAnalyzer
from graph.graph_attributes import GraphAttributes
from graph.graph_generator import GraphGenerator
from graph.graph_postprocessor import GraphPostProcessor
from graph.weight_length_mapping import mapping
from utils.debug_utils import debugging, timer
from utils.graph_utils import resize_image_to_fit_positions
from utils.plotting_utils import plot_graph
from utils.output_handler import OutputHandler  # Import OutputHandler


@debugging
@timer
def generate_and_process_graphs(
        set_name,
        resolution,
        config,
        num_iterations=300,
        graph_sample=10,
        save_plots=True,
        view_only=False,
        error_threshold=1.5,
        max_attempts=10
):
    output_handler = OutputHandler()  # Create an instance of OutputHandler

    if view_only:
        warnings.warn("[DEBUG] Only the original graph will be shown WITHOUT SAVING")
    else:
        save_dir = config.OUTPUT_DIR
        output_handler.archive_if_exists(save_dir)
        output_handler.ensure_directory(save_dir)
        print(f"Created new directory: {save_dir}")

    print(f"{'-' * 20} Processing {set_name}-{resolution} with {num_iterations} iterations {'-' * 20}")

    data_loader = DataLoader(config, set_name, resolution)
    data_loader.load_data()
    original_graph = data_loader.graph
    data_loader.image = resize_image_to_fit_positions(data_loader.image)
    original_graph_attributes = GraphAttributes(original_graph, config=config)

    mapped_weights, map_length_to_weight, length_bins, weight_baskets, y = mapping(original_graph)

    analyzer = GraphAnalyzer(original_graph)
    Q = [q / 100 for q in range(-300, 301, 10)]
    ori_tau_list = analyzer.calculate_multifractal_spectrum(Q)
    ori_alpha_0, ori_width, ori_al_list, ori_fal_list = analyzer.n_spectrum(ori_tau_list, Q)

    frame = [[0, config.DEFAULT_FRAME_RANGE], [0, config.DEFAULT_FRAME_RANGE]]

    original_graph_output_path = os.path.join(config.ORIGINAL_GRAPH_PATH, f"Set {set_name} Res {resolution}")
    synthetic_graph_output_path = os.path.join(config.SYNTHETIC_GRAPH_PATH, f"Set {set_name} Res {resolution}")

    output_handler.archive_if_exists(original_graph_output_path)
    output_handler.archive_if_exists(synthetic_graph_output_path)
    output_handler.ensure_directory(original_graph_output_path)
    output_handler.ensure_directory(synthetic_graph_output_path)

    # Plot original graph
    if view_only or save_plots:
        plot_graph(
            graph=original_graph,
            set_name=set_name,
            resolution=resolution,
            frame=frame,
            save=save_plots and not view_only,
            output_path=original_graph_output_path,
            background=True,
            image=data_loader.image,
            alpha=0.6,
            title='Original Graph'
        )
        cv2.imwrite(os.path.join(original_graph_output_path, f'Original Image - Set {set_name} Res {resolution}.png'),
                    data_loader.image)
        if view_only:
            return

    # Generate synthetic graphs
    synthetic_graphs = []
    errors = []

    total_iterations = num_iterations
    for i in tqdm(range(total_iterations), desc="Generating Graphs", ncols=80):
        error = float('inf')
        attempt = 0

        while error > error_threshold and attempt < max_attempts:
            attempt += 1

            # Generate synthetic graph
            generator = GraphGenerator(original_graph_attributes)
            synthetic_graph, frame = generator.generate_graph()

            # Post-process the graph
            postprocessor = GraphPostProcessor(synthetic_graph, original_graph_attributes)
            postprocessor.adjust_degree_distribution()
            postprocessor.assign_weights(map_length_to_weight, length_bins, weight_baskets, y)
            processed_graph = postprocessor.graph

            # Calculate multifractal properties
            analyzer = GraphAnalyzer(processed_graph)
            tau_list = analyzer.calculate_multifractal_spectrum(Q)
            alpha_0, width, al_list, fal_list = analyzer.n_spectrum(tau_list, Q)

            # Calculate error
            error = euclidean([ori_alpha_0, ori_width], [alpha_0, width])

        if attempt == max_attempts and error > error_threshold:
            print(f"Failed to generate a valid graph after {max_attempts} attempts.")
            continue  # Skip this iteration

        synthetic_graphs.append(processed_graph)
        errors.append(error)

        # Plot and save a sample of graphs
        if save_plots and i < graph_sample:
            plot_graph(
                graph=processed_graph,
                set_name=set_name,
                resolution=resolution,
                frame=frame,
                save=True,
                output_path=synthetic_graph_output_path,
                title=f"Synthetic Graph {i + 1}"
            )
            print(f"Plotted and saved synthetic graph {i + 1} with error {error:.4f}")
        else:
            print(f"Generated synthetic graph {i + 1} with error {error:.4f}")

    # Save synthetic graphs
    save_dir = os.path.join(config.OUTPUT_DIR, f"Set {set_name} Res {resolution}")
    output_handler.ensure_directory(save_dir)

    save_path = os.path.join(save_dir, 'synthetic_graphs.pkl')
    output_handler.save_pickle(synthetic_graphs, save_path)

    # Save errors
    errors_path = os.path.join(save_dir, 'synthetic_graph_errors.pkl')
    output_handler.save_pickle(errors, errors_path)

    print(f"Synthetic graphs and errors saved to {save_dir}")


if __name__ == '__main__':
    config = Config()
    set_names = ['D']  # Add more set names as needed
    resolutions = ['10kX']  # Add more resolutions as needed
    graph_sample = 10  # Number of synthetic graphs to plot and save
    num_iterations = 300  # Total synthetic graphs to generate
    save_plots = True
    view_only = False  # Set to True to only plot the original graph

    for set_name in set_names:
        for resolution in resolutions:
            generate_and_process_graphs(
                set_name,
                resolution,
                config=config,
                num_iterations=num_iterations,
                graph_sample=graph_sample,
                save_plots=save_plots,
                view_only=view_only
            )
