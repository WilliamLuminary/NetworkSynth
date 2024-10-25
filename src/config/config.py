import os


class Config:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

        self.BASE_DATA_PATH = os.path.join(project_root, 'data')

        self.BASE_INPUT_PATH = os.path.join(self.BASE_DATA_PATH, 'input')
        self.POSITIONS_DIR = os.path.join(self.BASE_INPUT_PATH, 'position')
        self.SPARSE_MATRICES_DIR = os.path.join(self.BASE_INPUT_PATH, 'sparse_matrices')
        self.IMAGES_DIR = os.path.join(self.BASE_INPUT_PATH, 'Original Graphs')

        self.BASE_OUTPUT_PATH = os.path.join(self.BASE_DATA_PATH, 'output')
        self.OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_PATH, 'Results')
        self.SYNTHETIC_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, 'Synthetic Graphs')
        self.ORIGINAL_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, 'Original Graphs')

        self.DEFAULT_FRAME_RANGE = 510
        self.CLOSED_NODES_FACTOR = 1.5
        self.CLOSED_EDGES_FACTOR = 1.0

        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(self.BASE_INPUT_PATH, exist_ok=True)
        os.makedirs(self.POSITIONS_DIR, exist_ok=True)
        os.makedirs(self.SPARSE_MATRICES_DIR, exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
        os.makedirs(self.BASE_OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.SYNTHETIC_GRAPH_PATH, exist_ok=True)
        os.makedirs(self.ORIGINAL_GRAPH_PATH, exist_ok=True)
