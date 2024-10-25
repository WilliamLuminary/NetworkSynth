# src/main.py

import os
import pickle
import warnings

from scipy.spatial.distance import euclidean
from tqdm import tqdm

from config.config import Config
from data.data_loader import DataLoader
from graph.graph_analyzer import GraphAnalyzer
from graph.graph_attributes import GraphAttributes
from graph.graph_generator import GraphGenerator
from graph.graph_postprocessor import GraphPostProcessor
from properties.weight_length_mapping import mapping
from utils.debug_utils import debugging, timer
from utils.plotting_utils import plot_graph


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
        max_attempts=1000
):
    if view_only:
        warnings.warn("[DEBUG] Only the original graph will be shown WITHOUT SAVING")
    else:
        # Optionally delete the existing output directory
        save_dir = config.OUTPUT_DIR  # Or another appropriate path
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
            print(f"Deleted existing directory: {save_dir}")

    print(f"{'-' * 20} Processing {set_name}-{resolution} with {num_iterations} iterations {'-' * 20}")

    # Load original graph
    data_loader = DataLoader(config, set_name, resolution)
    data_loader.load_data()
    original_graph = data_loader.graph

    # Create a GraphAttributes instance for the original graph
    original_graph_attributes = GraphAttributes(original_graph, config=config)

    # Mapping edge lengths to weights
    mapped_weights, map_length_to_weight, length_bins, weight_baskets, y = mapping(original_graph)

    # Calculate multifractal properties of the original graph
    analyzer = GraphAnalyzer(original_graph)
    Q = [q / 100 for q in range(-300, 301, 10)]
    ori_tau_list = analyzer.calculate_multifractal_spectrum(Q)
    ori_alpha_0, ori_width, ori_al_list, ori_fal_list = analyzer.n_spectrum(ori_tau_list, Q)

    # Prepare frame for plotting
    frame = [[0, config.DEFAULT_FRAME_RANGE], [0, config.DEFAULT_FRAME_RANGE]]

    # Plot original graph
    if view_only or save_plots:
        output_path = config.get_original_graph_output_path(set_name, resolution)
        plot_graph(
            graph=original_graph,
            set_name=set_name,
            resolution=resolution,
            frame=frame,
            save=save_plots and not view_only,
            output_path=output_path,
            background=True,
        )
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

            # Post-process graph
            postprocessor = GraphPostProcessor(synthetic_graph, original_graph_attributes)
            postprocessor.adjust_degree_distribution()
            postprocessor.assign_weights(map_length_to_weight, length_bins, weight_baskets, y)
            processed_graph = postprocessor.graph

            # Calculate multi-fractal properties
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
                output_path=config.SYNTHETIC_GRAPH_PATH,
                title=f"Synthetic Graph {i + 1}"
            )
            print(f"Plotted and saved synthetic graph {i + 1} with error {error:.4f}")
        else:
            print(f"Generated synthetic graph {i + 1} with error {error:.4f}")

    save_path = os.path.join(config.OUTPUT_DIR, 'synthetic_graphs.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(synthetic_graphs, f)
    print(f"Synthetic graphs saved to {save_path}")


if __name__ == '__main__':
    config = Config()
    set_names = ['A']
    resolutions = ['10kX']
    num_iterations = 300  # Total synthetic graphs to generate
    graph_sample = 10  # Number of synthetic graphs to plot and save
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
