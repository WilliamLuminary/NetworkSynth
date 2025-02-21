# src/properties/utils.py
import functools
import logging
import time
from functools import singledispatch
from typing import Tuple, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import ndarray
from scipy.sparse import spmatrix

from config import BaseConfig


def calculate_frame(graph: nx.Graph = None,
                    center_position: Union[tuple, list, ndarray] = None,
                    frame_range: Tuple[int, int] = None) -> tuple[tuple[float, float], tuple[float, float]]:
    frame_range = frame_range or BaseConfig.DEFAULT_FRAME_SIZE
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


@singledispatch
def build_graph(*args):
    raise TypeError(f"Unsupported input types: {tuple(type(arg) for arg in args)}")


@build_graph.register
def _(positions: np.ndarray, adjacency_matrix: spmatrix) -> nx.Graph:
    # noinspection PyUnresolvedReferences
    graph = nx.from_scipy_sparse_array(adjacency_matrix, edge_attribute='weight')
    for i, pos in enumerate(positions):
        graph.nodes[i]['pos'] = pos

    graph = largest_connected_component(graph)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
    return graph


@build_graph.register
def _(nodes: set, edges: set) -> nx.Graph:
    graph = nx.Graph()
    position_map = {node.position: node for node in nodes}

    for node in nodes:
        graph.add_node(position_map[node.position].id, pos=node.position)

    for edge in edges:
        u, v = position_map[edge[0]].id, position_map[edge[1]].id
        graph.add_edge(u, v)

    return graph


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
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


def finalize_plot(fig: plt.Figure, show: bool = False) -> np.ndarray:
    plt.tight_layout(pad=0)
    if show:
        plt.show()
    fig = figure_to_ndarray(fig)
    plt.close()
    return fig


def figure_to_ndarray(fig: plt.Figure, swap_channels: bool = False) -> ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image_array = np.asarray(buf)
    image_array = image_array[..., [2, 1, 0, 3]] if swap_channels else image_array
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
        graph = largest_connected_component(graph)
    return graph
