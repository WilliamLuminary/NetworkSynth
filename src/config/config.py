# src/config/config.py

import os


class Config:
    def __init__(self):
        self.BASE_DATA_PATH = 'data'
        self.BASE_RESULTS_PATH = os.path.join(self.BASE_DATA_PATH, 'Results')

        self.POSITIONS_DIR = os.path.join(self.BASE_DATA_PATH, 'position')
        self.SPARSE_MATRICES_DIR = os.path.join(self.BASE_DATA_PATH, 'sparse_matrices')
        self.IMAGES_DIR = os.path.join(self.BASE_DATA_PATH, 'Original Graphs')

        self.OUTPUT_DIR = self.BASE_RESULTS_PATH
        self.SYNTHETIC_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, 'Synthetic Graphs')
        self.ORIGINAL_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, 'Original Graphs')

        self.DEFAULT_FRAME_RANGE = 510
        self.CLOSED_NODES_FACTOR = 1.5
        self.CLOSED_EDGES_FACTOR = 1.0

        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.SYNTHETIC_GRAPH_PATH, exist_ok=True)
        os.makedirs(self.ORIGINAL_GRAPH_PATH, exist_ok=True)
