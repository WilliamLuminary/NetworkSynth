# src/data/graph_data_agent.py
import logging
import os
import re
from queue import Queue

import cv2
import networkx as nx
import numpy as np

from config.base import BaseConfig
from config.name_resolution_set import NameResolutionSet

logger = logging.getLogger(__name__)


class GraphDataAgent(BaseConfig):
    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.set_name, self.resolution = str(self.name_res_set.set_name), str(self.name_res_set.resolution)

        self.positions_of_nodes = None
        self.adjacency_matrix = None
        self.original_image = None
        self.original_graph = None

        self.attributes = None

        self.synthetic_graphs = Queue()

    def add_synthetic_graph(self, graph):
        self.synthetic_graphs.put(graph)

    def get_synthetic_graph(self):
        return self.synthetic_graphs.get() if not self.synthetic_graphs.empty() else None

    def load_data(self):
        logger.info(f"Loading data for {self.name_res_set}")
        self.positions_of_nodes = self._load_positions()
        self.adjacency_matrix = self._load_sparse_matrix()
        self.original_image = self._load_image()
        self.original_graph = self._create_graph()
        logger.info(f"Data successfully loaded for {self.name_res_set}")

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

    def _create_graph(self):
        if self.positions_of_nodes is None or self.adjacency_matrix is None:
            err_msg = "Positions or sparse matrix not loaded before original_graph creation."
            logger.error(err_msg)
            raise ValueError(err_msg)
        graph = nx.from_scipy_sparse_array(self.adjacency_matrix, edge_attribute='weight')
        for i, pos in enumerate(self.positions_of_nodes):
            graph.nodes[i]['pos'] = pos.astype(np.float64)

        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()
        graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
        logger.info(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

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
