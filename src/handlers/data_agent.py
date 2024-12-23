# src/data/data_agent.py
import logging
import os
import re
from queue import Queue
from typing import Optional

import cv2
import numpy as np

from config import Config, NameResolutionSet
from graph import GraphAttributes
from utils import build_graph_pos_and_adj_mat
from utils.network_utils import MapHandler

logger = logging.getLogger(__name__)


class DataAgent(Config):
    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.set_name, self.resolution = str(self.name_res_set.set_name), str(self.name_res_set.resolution)

        self.positions_of_nodes = None
        self.adjacency_matrix = None
        self.original_image = None
        self.original_graph = None

        self.attributes: Optional[GraphAttributes] = None
        self.map_handler = None

        self.synthetic_graphs = Queue()

    def load_data(self):
        logger.info(f"Loading data for {self.name_res_set}")
        self.positions_of_nodes = self._load_positions()
        self.adjacency_matrix = self._load_sparse_matrix()
        self.original_image = self._resize_image(self._load_image())
        self.original_graph = build_graph_pos_and_adj_mat((self.positions_of_nodes,
                                                           self.adjacency_matrix))
        self._transform_graph_positions(self.original_graph, rotation_deg=270)
        self.map_handler = MapHandler(self.original_graph)
        logger.info(f"Data successfully loaded for {self.name_res_set}")

    def add_synthetic_graph(self, graph):
        self.synthetic_graphs.put(graph)

    def get_synthetic_graph(self):
        return self.synthetic_graphs.get() if not self.synthetic_graphs.empty() else None

    def set_attributes(self, attributes):
        self.attributes = attributes

    def _load_positions(self):
        pattern = re.compile(
            rf"{re.escape(self.set_name)}_{re.escape(self.resolution)}.*\.npy",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            self.POSITION_DATA_DIR, pattern, f'positions_of_nodes for {self.set_name}'
        )
        logger.info(f"Positions file loaded: {file_path}")
        positions = np.load(file_path, allow_pickle=True)
        return positions

    def _load_sparse_matrix(self):
        pattern = re.compile(
            rf"sparse_matrices_{re.escape(self.resolution)}.*\.npz",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            self.ADJ_MATRIX_DATA_DIR, pattern, 'sparse matrix'
        )
        logger.info(f"Sparse matrix file loaded: {file_path}")
        matrix_data = np.load(file_path, allow_pickle=True)

        set_name = self.set_name
        if set_name == 'C':
            if 'C1' in matrix_data:
                set_name = 'C1'
            elif 'C' in matrix_data:
                set_name = 'C'
            else:
                available_keys = list(matrix_data.keys())
                err_msg = f"Neither 'C' nor 'C1' is found in sparse matrix data. Available sets: {available_keys}"
                logger.error(err_msg)
                raise KeyError(err_msg)

        if set_name not in matrix_data:
            available_keys = list(matrix_data.keys())
            err_msg = f"Set '{set_name}' not found in sparse matrix data. Available sets: {available_keys}"
            logger.error(err_msg)
            raise KeyError(err_msg)

        return matrix_data[set_name].item()

    def _load_image(self):
        images_path = os.path.join(self.IMAGES_DIR, self.set_name)
        if self.set_name == 'B':
            # noinspection PyTypeChecker
            images_path = os.path.join(images_path, '1811')
        pattern = re.compile(
            rf"\b{re.escape(self.resolution)}\b.*\.(tif|png|jpg)",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            images_path, pattern, f'image for {self.set_name}'
        )
        logger.info(f"Image file loaded: {file_path}")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            err_msg = f"Failed to load image from file: {file_path}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return image

    @staticmethod
    def _find_file_with_pattern(directory_path, pattern, details=''):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files_in_directory = os.listdir(directory_path)
        matched_files = [file_name for file_name in files_in_directory if pattern.search(file_name)]

        if len(matched_files) == 1:
            file_path = os.path.join(directory_path, matched_files[0])
            return file_path
        elif len(matched_files) > 1:
            raise FileExistsError(f"Multiple {details} files found: {matched_files}.")
        else:
            raise FileNotFoundError(f"No {details} file found in {directory_path}.")

    @staticmethod
    def _resize_image(image, target_size=Config.DEFAULT_FRAME_RANGE):
        _height, _width = image.shape
        _scaling_factor = target_size / max(_width, _height)

        _new_width = int(_width * _scaling_factor)
        _new_height = int(_height * _scaling_factor)
        resized_image = cv2.resize(image, (_new_width, _new_height))
        return resized_image

    @staticmethod
    def _transform_positions(positions, flip_x=False, flip_y=False, rotation_deg=0):
        c = Config.DEFAULT_FRAME_RANGE / 2
        positions -= c
        if flip_x:
            positions[:, 0] = -positions[:, 0]
        if flip_y:
            positions[:, 1] = -positions[:, 1]
        if rotation_deg == 90:
            x = -positions[:, 1].copy()
            positions[:, 1] = positions[:, 0]
            positions[:, 0] = x
        elif rotation_deg == 180:
            positions = -positions
        elif rotation_deg == 270:
            x = positions[:, 1].copy()
            positions[:, 1] = -positions[:, 0]
            positions[:, 0] = x
        positions += c
        return positions

    @staticmethod
    def _transform_graph_positions(graph, flip_x=False, flip_y=False, rotation_deg=0):
        c = Config.DEFAULT_FRAME_RANGE / 2
        for node, d in graph.nodes(data=True):
            p = np.array(d.get('pos', [0, 0]), dtype=np.float64)
            p -= c
            if flip_x:
                p[0] = -p[0]
            if flip_y:
                p[1] = -p[1]
            if rotation_deg == 90:
                px = -p[1]
                py = p[0]
                p[0], p[1] = px, py
            elif rotation_deg == 180:
                p = -p
            elif rotation_deg == 270:
                px = p[1]
                py = -p[0]
                p[0], p[1] = px, py
            p += c
            d['pos'] = p
