from typing import Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np

from config import BaseConfig
from config.enums import Mode
from utils import build_graph


def _resize_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or BaseConfig.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]
    scaling_factor = target_size / min(height, width)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def _trim_cv2_image(image: np.ndarray, frame_range: Tuple[int, int] = None) -> np.ndarray:
    frame_range = frame_range or BaseConfig.DEFAULT_FRAME_SIZE
    target_size = min(frame_range)
    height, width = image.shape[:2]

    bottom = min(height, target_size)
    right = min(width, target_size)
    image = image[:bottom, :right]
    return image


def _transform_graph_coordinates(network: nx.Graph, image_shape: Tuple, flip_x=False, flip_y=False,
                                 rotation_deg=270):
    c = (image_shape[0] / 2, image_shape[1] / 2)

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


class DataLoader:
    def __init__(self, mode, **kwargs):
        self._set_name = None
        self._resolution = None
        self._both_networks_path = None

        self._original_image = None

        self._original_network = None
        self._synthetic_networks = []

        if mode == Mode.Generate:
            assert 'set_name' in kwargs and 'resolution' in kwargs, "set_name and resolution are required."
            self._set_name = kwargs['set_name']
            self._resolution = kwargs['resolution']
        elif mode == Mode.Analyze:
            assert 'path' in kwargs, "path is required."
            self._both_networks_path = kwargs['path']
        elif mode == Mode.ATTR_GENERATE:
            assert 'path' in kwargs, "path is required."
            self._attr_path = kwargs['path']
            self._attr = None

        self._mode = mode

    def get_original_image(self) -> np.ndarray:
        return self._original_image

    def get_original_network(self) -> Union[nx.Graph, List[nx.Graph]]:
        return self._original_network

    def get_attr_dict(self) -> Dict:
        return self._attr

    def get_synthetic_networks(self) -> List[nx.Graph]:
        return self._synthetic_networks

    def add_synthetic_graph(self, graph: nx.Graph):
        self._synthetic_networks.append(graph)

    def load(self):
        if self._mode == Mode.Generate:
            self._original_network = build_graph(self._load_positions(), self._load_sparse_matrix())

            if not BaseConfig.DEFAULT_FRAME_SIZE:
                BaseConfig.update_frame_size((self._original_image.shape[1], self._original_image.shape[0]))

            image = _resize_cv2_image(self._original_image)
            image = _trim_cv2_image(image)
            self._original_image = image
            _transform_graph_coordinates(self._original_network, image.shape)

        elif self._mode == Mode.Analyze:
            self._original_network, self._synthetic_networks = self._load_networks()

        elif self._mode == Mode.ATTR_GENERATE:
            self._attr = self._load_attr()

    def _load_positions(self) -> Union[np.ndarray, List]:
        return BaseConfig.POSITION_DATA_FUNC(str(self._set_name), str(self._resolution))

    def _load_sparse_matrix(self) -> Union[np.ndarray, List]:
        return BaseConfig.ADJ_MATRIX_DATA_FUNC(str(self._set_name), str(self._resolution))

    def _load_image(self) -> Union[np.ndarray, List]:
        return BaseConfig.IMAGES_FUNC(str(self._set_name), str(self._resolution))

    def _load_networks(self) -> Tuple[List[nx.Graph], List[nx.Graph]]:
        return BaseConfig.NETWORKS_FUNC(self._both_networks_path)

    def _load_attr(self) -> Dict:
        return BaseConfig.ATTRIBUTES_DICT_FUNC(self._attr_path)
