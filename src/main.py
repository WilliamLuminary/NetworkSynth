# src/main.py

import os
import pickle
from config.config import Config
from data.data_loader import DataLoader
from graph.graph_attributes import GraphAttributes
from graph.graph_generator import GraphGenerator
from graph.graph_postprocessor import GraphPostProcessor
from graph.graph_analyzer import GraphAnalyzer
from properties.weight_length_mapping import mapping
from plotting.graph_plotter import plot_graph_with_positions
from utils.debug_utils import DEBUG, debugging, timer
from tqdm import tqdm


@debugging
@timer
def generate_and_process_graphs(set_name, resolution, config, num_iterations=10, save_plots=False):
    data_loader = DataLoader(config, set_name, resolution)
    data_loader.load_data()
    original_graph = data_loader.graph
    image = data_loader.image

    # Create GraphAttributes instance for the original graph
    original_graph_attributes = GraphAttributes(original_graph)

    # Mapping edge lengths to weights
    mapped_weights, map_length_to_weight, length_bins, weight_baskets, y = mapping(original_graph)

    # Plot original graph
    frame = [[0, config.DEFAULT_FRAME_RANGE], [0, config.DEFAULT_FRAME_RANGE]]
    plot_graph_with_positions(
        original_graph, frame, set_name=set_name, resolution=resolution,
        save=save_plots, base_path=config.OUTPUT_DIR, background=True
    )

    # Generate synthetic graphs
    synthetic_graphs = []
    for _ in tqdm(range(num_iterations)):
        generator = GraphGenerator(original_graph_attributes, config)
        synthetic_graph, frame = generator.generate_graph()

        postprocessor = GraphPostProcessor(synthetic_graph, original_graph_attributes)
        postprocessor.adjust_degree_distribution()
        postprocessor.assign_weights(map_length_to_weight, length_bins, weight_baskets, y)

        # Analyze and plot synthetic graph
        analyzer = GraphAnalyzer(postprocessor.graph)
        Q = [q / 100 for q in range(-300, 301, 10)]
        tau_list = analyzer.calculate_multifractal_spectrum(Q)
        alpha_0, width, al_list, fal_list = analyzer.n_spectrum(tau_list, Q)

        synthetic_graphs.append(postprocessor.graph)

        if save_plots:
            plot_graph_with_positions(
                postprocessor.graph, frame, set_name=set_name, resolution=resolution,
                save=True, base_path=config.OUTPUT_DIR
            )

    # Save synthetic graphs
    save_dir = os.path.join(config.OUTPUT_DIR, f'Set {set_name} Res {resolution}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'synthetic_graphs.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(synthetic_graphs, f)
    print(f"Synthetic graphs saved to {save_path}")


if __name__ == '__main__':
    config = Config()
    set_names = ['A']
    resolutions = ['10kX']
    num_iterations = 10
    save_plots = True

    for set_name in set_names:
        for resolution in resolutions:
            generate_and_process_graphs(set_name, resolution, config=config, num_iterations=num_iterations,
                                        save_plots=save_plots)
