import os

from . import Config, Resolution, SetName


class ConfigOld(Config):
    SETS = [SetName.A, SetName.B, SetName.C, SetName.D]
    RESOLUTIONS = [Resolution.X10K]

    DEFAULT_FRAME_SIZE = (510, 510)
    CLOSED_NODES_FACTOR = 1.5
    CLASSES_FACTOR = 1
    # For 10_kx image, node fac should be 1.5, and edge fac should be 1

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MEASURE_WEIGHTED = True

    BASE_INPUT_PATH = os.path.join(Config.BASE_DATA_PATH, 'input', 'old_input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')

    @classmethod
    def initialize(cls):
        cls._setup_logger(details="old")
