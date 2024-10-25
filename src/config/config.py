# config.py

import os


class Config:
    def __init__(self):
        # Base paths
        self.BASE_DATA_PATH = 'data'
        self.BASE_RESULTS_PATH = os.path.join(self.BASE_DATA_PATH, 'Results')

        # Input directories
        self.POSITIONS_DIR = os.path.join(self.BASE_DATA_PATH, 'position')
        self.SPARSE_MATRICES_DIR = os.path.join(self.BASE_DATA_PATH, 'sparse_matrices')
        self.IMAGES_DIR = os.path.join(self.BASE_DATA_PATH, 'Original Graphs')

        # Output directory
        self.OUTPUT_DIR = self.BASE_RESULTS_PATH

        # Other configurations
        self.DEFAULT_FRAME_RANGE = 510
        self.CLOSED_NODES_FACTOR = 1.5
        self.CLOSED_EDGES_FACTOR = 1.0

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(self.POSITIONS_DIR, exist_ok=True)
        os.makedirs(self.SPARSE_MATRICES_DIR, exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
