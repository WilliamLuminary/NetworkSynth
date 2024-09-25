import os

from src.preprocess.image_utils import create_graph, plot_network_with_graph
from src.preprocess.load_files import load_positions, load_sparse_matrix, find_image_file, load_image_file
from src.utils.debug_utils import debugging


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

    graph = create_graph(positions, sparse_matrix)
    if not load_image:
        return graph

    image_file = find_image_file(set_name, resolution, os.path.join(file_path, 'Original Graphs'))
    image = load_image_file(image_file)
    plot_network_with_graph(graph, image, set_name, resolution, save=save, base_path=save_path, background=background,
                            alpha=alpha)
    return graph
