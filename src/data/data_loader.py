# src/data/data_loader.py

import os
import re
import cv2
import numpy as np
import networkx as nx
from utils.debug_utils import debugging


class DataLoader:
    def __init__(self, config, set_name, resolution):
        self.config = config
        self.set_name = set_name
        self.resolution = resolution
        self.positions = None
        self.sparse_matrix = None
        self.image = None
        self.graph = None

    @debugging
    def load_data(self):
        self.positions = self._load_positions()
        self.sparse_matrix = self._load_sparse_matrix()
        self.image = self._load_image()
        self.graph = self._create_graph()

    def _load_positions(self):
        pattern = re.compile(
            rf"{re.escape(self.set_name)}_{re.escape(self.resolution)}.*(pos|position)\.npy",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            self.config.POSITIONS_DIR, pattern, f'positions for {self.set_name}'
        )
        positions = np.load(file_path, allow_pickle=True)
        return positions

    def _load_sparse_matrix(self):
        pattern = re.compile(
            rf"sparse_matrices_{re.escape(self.resolution)}.*\.npz",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            self.config.SPARSE_MATRICES_DIR, pattern, 'sparse matrix'
        )
        matrix_data = np.load(file_path, allow_pickle=True)

        set_name = self.set_name
        if set_name == 'C':
            if 'C1' in matrix_data:
                set_name = 'C1'
            elif 'C' in matrix_data:
                set_name = 'C'
            else:
                available_keys = list(matrix_data.keys())
                raise KeyError(f"Neither 'C' nor 'C1' is found. Available sets: {available_keys}")

        if set_name not in matrix_data:
            available_keys = list(matrix_data.keys())
            raise KeyError(f"Set '{set_name}' not found. Available sets: {available_keys}")

        return matrix_data[set_name].item()

    def _load_image(self):
        images_path = os.path.join(self.config.IMAGES_DIR, self.set_name)
        if self.set_name == 'B':
            images_path = os.path.join(images_path, '1811')
        pattern = re.compile(
            rf"\b{re.escape(self.resolution)}\b.*\.(tif|png|jpg)",
            re.IGNORECASE
        )
        file_path = self._find_file_with_pattern(
            images_path, pattern, f'image for {self.set_name}'
        )
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from file: {file_path}")
        return image

    def _create_graph(self):
        if self.positions is None or self.sparse_matrix is None:
            raise ValueError("Positions or sparse matrix not loaded.")
        graph = nx.from_scipy_sparse_array(self.sparse_matrix, edge_attribute='weight')
        for i, pos in enumerate(self.positions):
            graph.nodes[i]['pos'] = pos.astype(np.float64)

        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()
        graph = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
        return graph

    def _find_file_with_pattern(self, directory_path, pattern, details=''):
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
