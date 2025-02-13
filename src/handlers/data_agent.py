# src/data/data_agent.py
import logging
from typing import Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np

from config import Config, DataType
from graph import GraphAttrAgent
from utils import build_graph_pos_and_adj_mat, plot_graph
from .mapper import Mapper
from .saver import Saver

logger = logging.getLogger(__name__)


def _resize_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]
    scaling_factor = target_size / min(height, width)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def _trim_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]

    bottom = min(height, target_size)
    right = min(width, target_size)
    image = image[:bottom, :right]
    return image


def _transform_coordinates_by_image(network: nx.Graph, image: np.ndarray, flip_x=False, flip_y=False,
                                    rotation_deg=270):
    size = image.shape
    c = (size[0] / 2, size[1] / 2)

    if not flip_x and not flip_y and rotation_deg == 0:
        return

    for node, d in network.nodes(data=True):
        p = np.array(d.get('pos', [0, 0]), dtype=np.float64)

        p[0] -= c[0]
        p[1] -= c[1]

        if flip_x:
            p[0] = -p[0]
        if flip_y:
            p[1] = -p[1]

        if rotation_deg == 90:
            px, py = -p[1], p[0]
            p[0], p[1] = px, py
        elif rotation_deg == 180:
            p = -p
        elif rotation_deg == 270:
            px, py = p[1], -p[0]
            p[0], p[1] = px, py

        p[0] += c[0]
        p[1] += c[1]
        d['pos'] = p

    def save(self, data_type: DataType, file_name_prefix: str = None, arg=None):
        if not self.saver and data_type != DataType.SYNTHETIC_GRAPH:
            return

        show_figure_if_not_saving: bool = not self.saver and arg

        file_name_prefix = f"{file_name_prefix}_" or file_name_prefix
        if data_type == DataType.ORIGINAL_IMAGE:
            self.saver.save_file(self.original_image, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_NETWORK:
            self.saver.save_file(self.original_network, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_PROPERTY:
            self.saver.save_file(self.attributes, data_type, file_name_prefix)
        elif data_type == DataType.ORIGINAL_GRAPH:
            original_figure = plot_graph(
                data_type=DataType.ORIGINAL_GRAPH,
                graph=self.original_network,
                background=self.original_image,
                show=True
            )
            self.saver.save_file(original_figure, data_type, file_name_prefix)
        elif data_type == DataType.SYNTHETIC_GRAPH:
            assert isinstance(arg, nx.Graph), \
                "Content must be a networkx.Graph object and should be a synthetic network."
            synthetic_figure = plot_graph(
                data_type=DataType.SYNTHETIC_GRAPH,
                graph=arg,
                show=show_figure_if_not_saving
            )
            self.saver.save_file(synthetic_figure, DataType.SYNTHETIC_GRAPH, file_name_prefix)
            self.add_synthetic_graph(arg)
        elif data_type == DataType.SYNTHETIC_NETWORK:
            self.saver.save_file(self.synthetic_networks, data_type, file_name_prefix)
