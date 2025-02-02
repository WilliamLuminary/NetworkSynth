# src/data/data_agent.py
import logging
from typing import Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np

from config import Config
from graph import GraphAttrAgent
from handlers.mapper import Mapper
from utils import build_graph_pos_and_adj_mat

logger = logging.getLogger(__name__)


class DataAgent:
    def __init__(self, set_name, resolution):
        self.set_name, self.resolution = set_name, resolution

        self.positions_of_nodes = None
        self.adjacency_matrix = None
        self.original_image = None
        self.original_network = None

        self.attributes: Optional[GraphAttrAgent] = None
        self.mapper = None

        self.synthetic_networks = []

    def load_data(self):
        logger.info(f"Loading data for {self.set_name}, {self.resolution}...")
        self.positions_of_nodes = self._load_positions()
        self.adjacency_matrix = self._load_sparse_matrix()
        self.original_image = self._load_image()
        self._resize_image()
        self._trim_image()

        self.original_network = build_graph_pos_and_adj_mat((self.positions_of_nodes,
                                                             self.adjacency_matrix))
        self._transform_original_positions(rotation_deg=270)
        self.mapper = Mapper(self.original_network)

    def _load_positions(self) -> Union[np.ndarray, list]:
        return Config.POSITION_DATA_FUNC(str(self.set_name), str(self.resolution))

    def _load_sparse_matrix(self) -> Union[np.ndarray, list]:
        return Config.ADJ_MATRIX_DATA_FUNC(str(self.set_name), str(self.resolution))

    def _load_image(self) -> Union[np.ndarray, list]:
        return Config.IMAGES_FUNC(str(self.set_name), str(self.resolution))

    def add_synthetic_graph(self, graph: nx.Graph):
        """
        Add a synthetic graph to the list of synthetic graphs.
        NOT THREAD-SAFE.
        """
        # warn_mesg = 'Adding synthetic graphs is not thread-safe.
        # Use with caution.'
        # warnings.warn(warn_mesg)
        # logger.warning(warn_mesg)
        self.synthetic_networks.append(graph)

    # def get_synthetic_graph(self):
    #     return self.synthetic_networks.get() if not self.synthetic_networks.empty() else None

    def set_attributes(self, attributes: GraphAttrAgent):
        self.attributes = attributes

    def _resize_image(self, frame_range: Tuple[int, int] = None) -> None:
        frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
        if self.original_image is None:
            return
        target_size = min(frame_range)
        height, width = self.original_image.shape[:2]
        scaling_factor = target_size / min(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        self.original_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _trim_image(self, frame_range: Tuple[int, int] = None) -> None:
        frame_range = frame_range or Config.DEFAULT_FRAME_SIZE
        if self.original_image is None:
            return
        target_size = min(frame_range)
        height, width = self.original_image.shape[:2]

        bottom = min(height, target_size)
        right = min(width, target_size)
        self.original_image = self.original_image[:bottom, :right]

    def _transform_original_positions(self, flip_x=False, flip_y=False, rotation_deg=0):
        if self.original_image is None:
            return
        _graph = self.original_network
        __size = self.original_image.shape
        c = (__size[0] / 2, __size[1] / 2)

        if not flip_x and not flip_y and rotation_deg == 0:
            return
        for node, d in _graph.nodes(data=True):
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
