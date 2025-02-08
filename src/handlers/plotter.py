# src/utils/plotter.py
import logging
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import ndarray

from config import Config, DataType, FILE_CONFIGURATIONS, FileTag
from utils import build_graph_pos_and_adj_mat, calculate_frame
from utils.utils import figure_to_ndarray

logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, original_image: ndarray, original_graph: nx.Graph):
        self.original_graph = original_graph
        self.original_image = original_image

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
                    graph = self.original_graph
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
            if self.original_image is None:
                logger.info("No background has been provided.")
            else:
                image = self.original_image
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
        graph = figure_to_ndarray(fig)
        plt.close()
        return graph
