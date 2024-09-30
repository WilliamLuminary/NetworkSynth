import os
import pickle
import shutil
import warnings

from scipy.spatial.distance import euclidean
from tqdm.notebook import tqdm

from src.genration.network_generation import plot_and_generate_network
from src.postprocess.assign_weight import process_weighted_graph, plot_graph_with_positions
from src.postprocess.multifractal_analysis import wnfd_nk, n_spectrum
from src.postprocess.reduce_node_edge import adjust_degree_distribution
from src.preprocess.raw_files_process import load_and_plot_graph
from src.properties.graph_matrics import prepare_graph_metrics
from src.properties.graph_properties import prepare_graph_properties
from src.properties.weight_length_mapping import mapping
from src.utils.debug_utils import debugging, timer, DEBUG


@debugging
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}", debug=True)


@debugging
def delete_existing_folder(set_name, resolution, base_path='/content/drive/MyDrive/vis/Results Yaxing'):
    folder_path = os.path.join(base_path, f"Set {set_name} Res {resolution}")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}", debug=True)


def calculate_graph_properties(graph, Q):
    ntau = wnfd_nk(graph, Q, weight=True, draw=False)
    alpha_0, width, al_list, fal_list = n_spectrum(ntau, Q)
    return alpha_0, width, al_list, fal_list


@debugging
@timer
def generate_and_process_graphs(set_name, resolution, closed_nodes_factor=1.5,
                                closed_edges_factor=1, graph_sample=10, num_iterations=300,
                                base_path='/content/drive/MyDrive/vis/Results Yaxing',
                                view_only=False):  # view_only means only the original graph will be plotted, it will not be saved.
    """
    Generate and process graphs for a given set and resolution.
    :param base_path:
    :param closed_edges_factor:
    :param closed_nodes_factor:
    :param resolution:
    :param set_name:
    :param view_only: When this value is True, only the original graph will be plotted without saving. For debugging!
    :param graph_sample: When this value is greater than 0, graph_sample graphs will be generated, plotted, and saved.
    :param num_iterations: When this value is greater than 0, num_iterations graphs will be generated and saved.
    ...
    """
    saved: bool = False
    if view_only:
        warnings.warn("[DEBUG] Only the original graph will be shown WITHOUT SAVING")
    else:
        saved = True
        delete_existing_folder(set_name, resolution, base_path=base_path)

    print(f"{'-' * 20} Processing {set_name}-{resolution} with {num_iterations} iterations {'-' * 20}")

    ori_G = load_and_plot_graph(set_name, resolution, save_path=base_path, save=saved, alpha=0.6)
    if view_only:
        return

    ori_G.number_of_nodes()
    ori_G.number_of_edges()
    degree_distribution, degree_transition_probs, degree_angles, degree_edge_lengths, avg_degree, avg_length = prepare_graph_properties(
        ori_G)
    mapped_weights, map_length_to_weight, length_bins, weight_baskets, y = mapping(ori_G, plot=False)
    range_of_q = 300
    Q = [q / 100 for q in range(-range_of_q, range_of_q + 1, 10)]
    ori_alpha_0, ori_width, ori_al_list, ori_fal_list = calculate_graph_properties(ori_G, Q)
    ori_graph_metrics = prepare_graph_metrics(ori_G)

    total_iterations = max(graph_sample, num_iterations)
    G_weighted_list, G_reduced_weighted_list, errors = [], [], []
    for i in tqdm(range(total_iterations)):

        max_attempts = 1000
        for attempt in range(max_attempts):
            syn_G, frame = plot_and_generate_network(
                degree_distribution=degree_distribution,
                degree_transition_probs=degree_transition_probs,
                degree_angles=degree_angles,
                degree_edge_lengths=degree_edge_lengths,
                closed_range=avg_length,
                closed_edges_factor=closed_edges_factor,
                closed_nodes_factor=closed_nodes_factor,
                skip_plotting=not DEBUG  # This is not the final graph, skip
            )

            weighted_graph = process_weighted_graph(syn_G, frame, map_length_to_weight, length_bins, weight_baskets, y,
                                                    "", save=False, skip_plotting=True)

            reduced_graph = adjust_degree_distribution(syn_G, ori_graph_metrics)
            reduced_weighted_graph = process_weighted_graph(reduced_graph, frame, map_length_to_weight, length_bins,
                                                            weight_baskets, y, "", save=False, skip_plotting=True)
            reduced_alpha_0, reduced_width, _, _ = calculate_graph_properties(reduced_weighted_graph, Q)
            alpha_0, width, _, _ = calculate_graph_properties(weighted_graph, Q)

            error = euclidean([ori_alpha_0, 0], [alpha_0, 0])  # We use reduced results by default
            if error < 1.5:
                break
            else:
                print(f"Aborting non-related network after {attempt} attempt(s)", debug=True)
        else:
            raise ValueError(f"Failed to generate a graph within {max_attempts} attempts.")
            # print(f"Failed to generate a graph within {max_attempts} attempts.")

        G_weighted_list.append(weighted_graph)
        G_reduced_weighted_list.append(reduced_weighted_graph)

        if i < graph_sample:
            plot_graph_with_positions(reduced_weighted_graph, "", frame, save=True, base_path=base_path)
            print(
                f'Nodes in final graph: {len(reduced_weighted_graph.nodes())}, edges: {len(reduced_weighted_graph.edges())}')
        else:
            print(
                f'Nodes in final graph: {len(reduced_weighted_graph.nodes())}, edges: {len(reduced_weighted_graph.edges())}')

    save_dir = f'{base_path}/Set {set_name} Res {resolution}'
    ensure_directory_exists(save_dir)

    save_path_weighted = os.path.join(save_dir,
                                      f'G_weighted_list Num {total_iterations} - Set {set_name} Res {resolution}.pkl')
    with open(save_path_weighted, 'wb') as file:
        pickle.dump(G_weighted_list, file)
    print(f"G_weighted_list has been saved to {save_path_weighted}")

    save_path_reduced_weighted = os.path.join(save_dir,
                                              f'G_reduced_weighted_list Num {total_iterations} - Set {set_name} Res {resolution}.pkl')
    with open(save_path_reduced_weighted, 'wb') as file:
        pickle.dump(G_reduced_weighted_list, file)
    print(f"G_reduced_weighted_list has been saved to {save_path_reduced_weighted}")
