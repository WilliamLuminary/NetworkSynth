import os

from config import Config, Resolution, SetName


class ConfigNew(Config):
    SETS = [SetName.S4, SetName.S8, SetName.S11, SetName.S14, SetName.S17, SetName.S20, SetName.S23, SetName.S26,
            SetName.S29, SetName.S32]
    RESOLUTIONS = [Resolution.NA]

    DEFAULT_FRAME_SIZE = (470, 470)
    CLOSED_NODES_FACTOR = 1.2
    CLASSES_FACTOR = 0.8
    # For 10_kx image, node fac should be 1.5, and edge fac should be 1

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MEASURE_WEIGHTED = True

    BASE_INPUT_PATH = os.path.join(Config.BASE_DATA_PATH, 'input', 'new_input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')



    @classmethod
    def initialize(cls):
        cls._setup_logger(details="new")
