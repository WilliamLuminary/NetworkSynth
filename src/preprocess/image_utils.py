import os

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.utils.debug_utils import debugging


@debugging
def create_graph(positions, sparse_matrix):
    graph = nx.from_scipy_sparse_array(sparse_matrix, edge_attribute='weight')

    for i, pos in enumerate(positions):
        graph.nodes[i]['pos'] = pos.astype(np.float64)

    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
    return graph


def resize_image_to_fit_positions(image, target_size=510):
    original_height, original_width = image.shape
    scaling_factor = target_size / max(original_width, original_height)

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor


@debugging
def plot_network_with_graph(graph, image, set_name, resolution,
                            save=False, base_path='/content/drive/MyDrive/vis/Results Yaxing', background=True,
                            alpha: float = 1):
    positions = np.array([graph.nodes[node]['pos'] for node in graph.nodes()])
    positions[:, [1, 0]] = positions[:, [0, 1]]
    positions[:, 1] = 510 - positions[:, 1]

    resized_image, scaling_factor = resize_image_to_fit_positions(image)
    image_extent = [0, resized_image.shape[1], 0, resized_image.shape[0]]

    plt.figure(figsize=(10, 10))
    if background:
        plt.imshow(resized_image, cmap='gray', extent=image_extent, alpha=alpha)

    for u, v in graph.edges():
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
