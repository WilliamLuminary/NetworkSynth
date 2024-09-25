import os

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.io_utils import load_positions, load_sparse_matrix, find_image_file, load_image_file
from src.utils import debugging


@debugging
def create_graph(positions, sparse_matrix):
    G = nx.from_scipy_sparse_array(sparse_matrix, edge_attribute='weight')

    for i, pos in enumerate(positions):
        G.nodes[i]['pos'] = pos.astype(np.float64)

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    return G


def resize_image_to_fit_positions(image, target_size=510):
    original_height, original_width = image.shape
    scaling_factor = target_size / max(original_width, original_height)

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor


@debugging
def plot_network_with_graph(G, image, set_name, resolution,
                            save=False, base_path='/content/drive/MyDrive/vis/Results Yaxing', background=True,
                            alpha: float = 1):
    positions = np.array([G.nodes[node]['pos'] for node in G.nodes()])
    positions[:, [1, 0]] = positions[:, [0, 1]]
    positions[:, 1] = 510 - positions[:, 1]

    resized_image, scaling_factor = resize_image_to_fit_positions(image)
    image_extent = [0, resized_image.shape[1], 0, resized_image.shape[0]]

    plt.figure(figsize=(10, 10))
    if background:
        plt.imshow(resized_image, cmap='gray', extent=image_extent, alpha=alpha)

    for u, v in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=3)
    plt.plot(positions[:, 0], positions[:, 1], 'bo', markersize=3.5)

    # plt.title(f'Original Graph - Set {set_name} Res {resolution}')
    plt.axis('off')
    plt.tight_layout()

    if save:
        folder_path = os.path.join(base_path, f"Set {set_name} Res {resolution}")
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f'Original Graph - Set {set_name} Res {resolution} - Alpha {alpha}.png'))
        cv2.imwrite(os.path.join(folder_path, f'Original Image - Set {set_name} Res {resolution}.png'), resized_image)
        plt.close()
    else:
        plt.show()


@debugging
def load_and_plot_graph(set_name, resolution, file_path='/content/drive/MyDrive/vis',
                        save_path='/content/drive/MyDrive/vis/Results Yaxing', load_image=True, save=False,
                        background=True, alpha: float = 1):
    positions = load_positions(set_name, resolution, file_path)
    sparse_matrix = load_sparse_matrix(resolution, file_path, set_name)

    num_positions = len(positions)
    num_nodes = sparse_matrix.shape[0]
    if num_positions != num_nodes:
        raise ValueError(
            f"Mismatch between number of positions ({num_positions}) and number of nodes in the sparse matrix ({num_nodes})")

    G = create_graph(positions, sparse_matrix)
    if not load_image:
        return G

    image_file = find_image_file(set_name, resolution, os.path.join(file_path, 'Original Graphs'))
    image = load_image_file(image_file)
    plot_network_with_graph(G, image, set_name, resolution, save=save, base_path=save_path, background=background,
                            alpha=alpha)
    return G
