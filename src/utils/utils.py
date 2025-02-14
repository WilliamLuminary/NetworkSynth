# src/properties/utils.py
import functools
import logging
import time
from typing import Tuple, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import ndarray

from config import Config, DataType, FILE_CONFIGURATIONS, FileTag

logger = logging.getLogger(__name__)


def calculate_frame(graph: nx.Graph = None, center_position: Union[tuple, list, ndarray] = None,
                    frame_range: Tuple[int, int] = None) -> tuple[
    tuple[float, float], tuple[float, float]]:
    frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
    if graph is None:
        if not center_position:
            raise ValueError("Either synthetic_graph or center_position must be provided.")
        _center_x, _center_y = center_position
    else:
        if center_position:
            raise ValueError("Only synthetic_graph or center_position must be provided.")
        _positions = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
        _center_x, _center_y = _positions[:, 0].mean(), _positions[:, 1].mean()

    _half_range = (frame_range[0] / 2, frame_range[1] / 2)
    _width, _height = _half_range

    frame = (
        (round(_center_x - _width, 2), round(_center_x + _width, 2)),  # Horizontal range (width)
        (round(_center_y - _height, 2), round(_center_y + _height, 2))  # Vertical range (height)
    )
    return frame


def build_graph_pos_and_adj_mat(pos_and_adj_mat: tuple) -> nx.Graph:
    """
    :param pos_and_adj_mat: A tuple of positions and adjacency matrix.
    :return:
    """
    positions_of_nodes, adjacency_matrix = pos_and_adj_mat
    # noinspection PyUnresolvedReferences
    graph = nx.from_scipy_sparse_array(adjacency_matrix, edge_attribute='weight')
    for i, pos in enumerate(positions_of_nodes):
        graph.nodes[i]['pos'] = pos.astype(np.float64)

    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
    return graph


def build_graph_nodes_and_edges(nodes: Union[list, set], edges: Union[list, set]) -> nx.Graph:
    graph = nx.Graph()
    position_map = {node.position: node for node in nodes}
    for node in nodes:
        graph.add_node(position_map[node.position].id, pos=node.position)

    for edge in edges:
        u, v = position_map[edge[0]].id, position_map[edge[1]].id
        graph.add_edge(u, v)
    return graph


def keep_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """
    Keep only the largest connected component of the graph.
    :param graph: A networkx graph, possibly with multiple connected components.
    :return: A networkx graph with only the largest connected component.

    Post condition:
        - The original graph remains unchanged.
        - The returned graph is nx.graph copy of the largest connected component.
    """
    if graph.number_of_nodes() >= 1:
        return graph.copy()

    if nx.is_connected(graph):
        return graph.copy()

    largest_cc = max(nx.connected_components(graph), key=len)
    # noinspection PyTypeChecker
    return graph.subgraph(largest_cc).copy()


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Time taken by func \"{func.__name__}\" : {round(end - start, 2)} seconds")
        return result

    return wrapper


def plot_network(data_type: DataType,
                 graph: nx.Graph = None,
                 pos_and_adj_mat: Union[tuple, list, ndarray] = None,
                 adjust_axis: bool = False, **kwargs) -> ndarray:
    """
    :param data_type:
    :param graph: If provided, plot the graph directly.
    :param pos_and_adj_mat: Only used if `graph` is not provided.
    Tuple / List / Ndarray of positions and adjacency matrix.
    :param adjust_axis:
    :param kwargs: Title, frame, plot_in_frame, background, image, alpha, line_width, node_size, output_path
    """
    if not data_type.has_tag(FileTag.PLOT):
        raise ValueError(f"Invalid data type: {data_type}, only graphs can be passed to this method.")
    if graph is None:
        if pos_and_adj_mat is None:
            raise ValueError("Either `graph` or `position and adj matrix` must be provided.")
        else:
            graph = build_graph_pos_and_adj_mat(pos_and_adj_mat)

    file_config = FILE_CONFIGURATIONS.get(data_type)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    position_dict = nx.get_node_attributes(graph, 'pos')
    if adjust_axis and position_dict is not None:
        positions_array = np.array([position_dict[node] for node in graph.nodes()])
        positions_array[:, [1, 0]] = positions_array[:, [0, 1]]
        positions_array[:, 1] = Config.DEFAULT_FRAME_SIZE - positions_array[:, 1]
        position_dict = {node: pos for node, pos in zip(graph.nodes(), positions_array)}

    line_width = file_config.line_width
    for u, v in graph.edges():
        pos_u = position_dict.get(u)
        pos_v = position_dict.get(v)
        if pos_u is not None and pos_v is not None:
            x_values = [pos_u[0], pos_v[0]]
            y_values = [pos_u[1], pos_v[1]]
            ax.plot(x_values, y_values, 'r-', linewidth=line_width, zorder=2)

    node_size = file_config.node_size
    for node in graph.nodes():
        pos = position_dict.get(node)
        if pos is not None:
            ax.plot(pos[0], pos[1], 'bo', markersize=node_size, zorder=2)

    # _frame = calculate_frame(graph)
    if data_type is DataType.ORIGINAL_GRAPH:
        frame = getattr(file_config, 'frame',
                        ((0, Config.DEFAULT_FRAME_SIZE[0]), (0, Config.DEFAULT_FRAME_SIZE[1])))
    else:
        frame = calculate_frame(graph)

    ax.set_xlim(frame[0])
    ax.set_ylim(frame[1])

    if data_type is DataType.ORIGINAL_GRAPH:
        if kwargs.get('background', None) is None:
            logger.info("No background has been provided.")
        else:
            image = kwargs['background']
            image_extent = (0, image.shape[1], 0, image.shape[0])
            alpha = getattr(file_config, 'alpha', 1.0)
            ax.imshow(image, cmap='gray', extent=image_extent, alpha=alpha)
    else:
        ax.add_patch(
            plt.Rectangle(
                (frame[0][0], frame[1][0]),
                frame[0][1] - frame[0][0],
                frame[1][1] - frame[1][0],
                facecolor='none',
                edgecolor=(0, 0, 0, 0.8),
                linewidth=2,
                zorder=1
            ))

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout(pad=0)

    if getattr(file_config, 'show_on_the_fly', False):
        plt.show()
    fig = figure_to_ndarray(fig)
    plt.close()
    return fig


def figure_to_ndarray(fig: plt.Figure, swap: bool = False) -> ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image_array = np.asarray(buf)
    image_array = image_array[..., [2, 1, 0, 3]] if swap else image_array
    return image_array


def trim_graph(graph: nx.Graph, tar_avg_deg: float) -> nx.Graph:
    """
    Trim the synthetic graph to have an average degree close to the original graph.
    :param graph: A synthetic graph.
    :param tar_avg_deg: The target average degree.
    :return: A copy of the trimmed synthetic graph.
    """
    while 2 * graph.number_of_edges() / graph.number_of_nodes() > 1.1 * tar_avg_deg:
        highest_degree_node = max(graph.degree, key=lambda x: x[1])[0]
        neighbors = list(graph.neighbors(highest_degree_node))
        if neighbors:
            graph.remove_edge(highest_degree_node, neighbors[0])
        graph = keep_largest_connected_component(graph)
    return graph
