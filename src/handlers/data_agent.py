# src/data/data_agent.py
import logging
import os
import re
from typing import Optional, Tuple

import cv2
import networkx as nx
import numpy as np

from config import Config, NameResolutionSet
from graph import GraphAttrAgent
from handlers.mapper import Mapper
from utils import build_graph_pos_and_adj_mat

logger = logging.getLogger(__name__)


class DataAgent(Config):
    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.set_name, self.resolution = str(self.name_res_set.set_name), str(self.name_res_set.resolution)

        self.positions_of_nodes = None
        self.adjacency_matrix = None
        self.original_image = None
        self.original_graph = None

        self.attributes: Optional[GraphAttrAgent] = None
        self.mapper = None

        self.synthetic_graphs = []

    def load_data(self):
        logger.info(f"Loading data for {self.name_res_set}")
        self.positions_of_nodes = self._load_positions()
        self.adjacency_matrix = self._load_sparse_matrix()

        self._load_image()
        self._resize_image()
        self._trim_image()

        self.original_graph = build_graph_pos_and_adj_mat((self.positions_of_nodes,
                                                           self.adjacency_matrix))
        self._transform_original_positions(rotation_deg=270)
        # self._transform_original_positions(rotation_deg=270)
        self.mapper = Mapper(self.original_graph)
        logger.info(f"Data successfully loaded for {self.name_res_set}")

    def add_synthetic_graph(self, graph: nx.Graph):
        """
        Add a synthetic graph to the list of synthetic graphs.
        NOT THREAD-SAFE.
        """
        # warn_mesg = 'Adding synthetic graphs is not thread-safe.
        # Use with caution.'
        # warnings.warn(warn_mesg)
        # logger.warning(warn_mesg)
        self.synthetic_graphs.append(graph)

    # def get_synthetic_graph(self):
    #     return self.synthetic_graphs.get() if not self.synthetic_graphs.empty() else None

    def set_attributes(self, attributes):
        self.attributes = attributes

    def _load_positions(self):
        directory_path = os.path.join(self.POSITION_DATA_DIR, self.resolution)
        set_name_value = str(self.name_res_set.set_name)  # Convert SetName enum to its string value
        pattern = re.compile(
            rf"W-\d+-\d+-\d+_{re.escape(set_name_value)}_postion\.npy",
            re.IGNORECASE
        )

        # Ensure a single file is returned
        file_path = self._find_file_with_pattern(
            directory_path, pattern, f'positions_of_nodes for {self.set_name}'
        )
        if isinstance(file_path, list):
            file_path = file_path[0]  # Get the first file (it should always be a single match)

        logger.info(f"Positions file loaded: {file_path}")
        positions = np.load(file_path, allow_pickle=True)
        return positions

    def _load_sparse_matrix(self):
        directory_path = os.path.join(self.ADJ_MATRIX_DATA_DIR, self.resolution)
        file_name = "sparse_matrices.npz"
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sparse matrix file not found: {file_path}")

        matrix_data = np.load(file_path, allow_pickle=True)

        set_name_value = str(self.name_res_set.set_name)
        key_pattern = rf"W-\d+-\d+-\d+_{re.escape(set_name_value)}_EL"

        matching_keys = [key for key in matrix_data.keys() if re.fullmatch(key_pattern, key)]

        if len(matching_keys) == 1:
            logger.info(f"Sparse matrix loaded: {matching_keys[0]}")
            return matrix_data[matching_keys[0]].item()
        elif len(matching_keys) > 1:
            raise FileExistsError(f"Multiple matching sparse matrices found: {matching_keys}")
        else:
            available_keys = list(matrix_data.keys())
            raise KeyError(f"No matching sparse matrix found for set '{self.name_res_set.set_name}'. "
                           f"Available keys: {available_keys}")

    # def _load_positions(self):
    #     pattern = re.compile(
    #         rf"{re.escape(self.set_name)}_{re.escape(self.resolution)}.*\.npy",
    #         re.IGNORECASE
    #     )
    #     file_path = self._find_file_with_pattern(
    #         self.POSITION_DATA_DIR, pattern, f'positions_of_nodes for {self.set_name}'
    #     )
    #     logger.info(f"Positions file loaded: {file_path}")
    #     positions = np.load(file_path, allow_pickle=True)
    #     return positions
    #
    # def _load_sparse_matrix(self):
    #     pattern = re.compile(
    #         rf"sparse_matrices_{re.escape(self.resolution)}.*\.npz",
    #         re.IGNORECASE
    #     )
    #     file_path = self._find_file_with_pattern(
    #         self.ADJ_MATRIX_DATA_DIR, pattern, 'sparse matrix'
    #     )
    #     logger.info(f"Sparse matrix file loaded: {file_path}")
    #     matrix_data = np.load(file_path, allow_pickle=True)
    #
    #     set_name = self.set_name
    #     if set_name == 'C':
    #         if 'C1' in matrix_data:
    #             set_name = 'C1'
    #         elif 'C' in matrix_data:
    #             set_name = 'C'
    #         else:
    #             available_keys = list(matrix_data.keys())
    #             err_msg = f"Neither 'C' nor 'C1' is found in sparse matrix data. Available sets: {available_keys}"
    #             logger.error(err_msg)
    #             raise KeyError(err_msg)
    #
    #     if set_name not in matrix_data:
    #         available_keys = list(matrix_data.keys())
    #         err_msg = f"Set '{set_name}' not found in sparse matrix data. Available sets: {available_keys}"
    #         logger.error(err_msg)
    #         raise KeyError(err_msg)
    #
    #     return matrix_data[set_name].item()

    def _load_image(self):
        directory_path = os.path.join(self.IMAGES_DIR, self.resolution)
        set_name_value = str(self.name_res_set.set_name)  # Convert SetName enum to string

        # pattern = re.compile(
        #     rf"W-\d+-\d+-\d+_{re.escape(set_name_value)}_.*\.(tif|png|jpg)",
        #     re.IGNORECASE
        # )
        pattern = re.compile(
            rf"W-\d+-\d+-\d+_{re.escape(set_name_value)}\.(tif|png|jpg)",
            re.IGNORECASE
        )

        file_path = self._find_file_with_pattern(
            directory_path, pattern, f'image for {self.name_res_set.set_name}'
        )
        if file_path is None:
            logger.warning(f"Background image is None.")
            return

        logger.info(f"Image file loaded: {file_path}")

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            err_msg = f"Failed to load image from file: {file_path}"
            logger.warning(err_msg)
            self.original_image = None
            return

        self.original_image = image

    @staticmethod
    def _find_file_with_pattern(directory_path, pattern, details='', must_exist=True) -> Optional[str]:
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
            if must_exist:
                raise FileNotFoundError(f"No {details} file found in {directory_path}.")
            else:
                logger.warning(f"No {details} file found in {directory_path}.")
                return None

    def _resize_image(self, frame_range: Tuple[int, int] = Config.DEFAULT_FRAME_SIZE):
        if self.original_image is None:
            return
        target_size = min(frame_range)
        height, width = self.original_image.shape[:2]
        scaling_factor = target_size / min(height, width)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        self.original_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def _trim_image(self, frame_range: Tuple[int, int] = Config.DEFAULT_FRAME_SIZE):
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
        _graph = self.original_graph
        __size = self.original_image.shape
        c = (__size[0] / 2, __size[1] / 2)

        # if rotation_deg != 0 and c[0] != c[1]:
        #     logger.warning("Only square frames can be rotated. Abort!")
        #     return

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
