# src/properties/utils.py
import functools
import logging
import time
from typing import Tuple, Union

import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import ndarray

from config import Config


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


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Time taken by func \"{func.__name__}\" : {round(end - start, 2)} seconds")
        return result

    return wrapper


def figure_to_ndarray(fig: Figure) -> ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image_array = np.asarray(buf)
    return image_array


def keep_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """
    Keep only the largest connected component of the graph.
    :param graph: A networkx graph, possibly with multiple connected components.
    :return: A networkx graph with only the largest connected component.

    Post condition:
        - The original graph remains unchanged.
        - The returned graph is nx.graph copy of the largest connected component.
    """
    if graph.number_of_nodes() == 0:
        return graph
    largest_cc = max(nx.connected_components(graph), key=len)
    # noinspection PyTypeChecker
    return graph.subgraph(largest_cc).copy()
