# src/utils/plot_agent.py

from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import ndarray

from config import Config, DataType, FileTag
from utils import build_graph_pos_and_adj_mat, calculate_frame
from .data_agent import DataAgent


class PlotAgent(Config):
    def __init__(self, graph_data_agent: DataAgent):
        self.graph_data_agent = graph_data_agent

    def plot_graph(self, data_type: DataType, graph: nx.Graph = None,
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
                if data_type is DataType.SYNTHETIC_GRAPH:
                    raise ValueError("Either `graph` or `position and adj matrix` must be provided.")
                else:
                    graph = self.graph_data_agent.original_graph
            else:
                graph = build_graph_pos_and_adj_mat(pos_and_adj_mat)

        _file_config = Config.FILE_CONFIGURATIONS.get(data_type)
        _fig, _ax = plt.subplots(figsize=(10, 10), dpi=260)

        _position_dict = nx.get_node_attributes(graph, 'pos')
        if adjust_axis and _position_dict is not None:
            __positions_array = np.array([_position_dict[node] for node in graph.nodes()])
            __positions_array[:, [1, 0]] = __positions_array[:, [0, 1]]
            __positions_array[:, 1] = Config.DEFAULT_FRAME_RANGE - __positions_array[:, 1]
            _position_dict = {node: pos for node, pos in zip(graph.nodes(), __positions_array)}

        _line_width = _file_config.line_width
        for u, v in graph.edges():
            pos_u = _position_dict.get(u)
            pos_v = _position_dict.get(v)
            if pos_u is not None and pos_v is not None:
                __x_values = [pos_u[0], pos_v[0]]
                __y_values = [pos_u[1], pos_v[1]]
                _ax.plot(__x_values, __y_values, 'r-', linewidth=_line_width, zorder=2)

        _node_size = _file_config.node_size
        for node in graph.nodes():
            pos = _position_dict.get(node)
            if pos is not None:
                _ax.plot(pos[0], pos[1], 'bo', markersize=_node_size, zorder=2)

        _frame = calculate_frame(graph)
        if 'frame' in kwargs:
            _frame = kwargs['frame']
        _ax.set_xlim(_frame[0])
        _ax.set_ylim(_frame[1])

        if data_type is DataType.ORIGINAL_GRAPH:
            _image = self.graph_data_agent.original_image
            _image_extent = (0, _image.shape[1], 0, _image.shape[0])
            _alpha = kwargs['alpha'] if 'alpha' in kwargs else 1
            _ax.imshow(_image, cmap='gray', extent=_image_extent, alpha=_alpha)
        else:
            _ax.add_patch(
                plt.Rectangle(
                    (_frame[0][0], _frame[1][0]),
                    _frame[0][1] - _frame[0][0],
                    _frame[1][1] - _frame[1][0],
                    facecolor='none',
                    edgecolor=(0, 0, 0, 0.8),
                    linewidth=2,
                    zorder=1
                ))

        if 'title' in kwargs:
            plt.title(kwargs['title'])

        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.axis('off')

        plt.tight_layout(pad=0)

        if 'show' in kwargs and kwargs['show']:
            plt.show()
        _graph = self._figure_to_ndarray_direct(plt)
        plt.close()
        return _graph

    @staticmethod
    def _figure_to_ndarray_direct(fig: plt) -> ndarray:
        _canvas = FigureCanvas(fig)
        _canvas.draw()
        _buf = _canvas.buffer_rgba()
        _image = np.asarray(_buf)
        return _image
