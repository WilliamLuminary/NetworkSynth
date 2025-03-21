# src/config/generate_mode/config_linlin.py
import logging
import os

import numpy as np

from ..base_config import BaseConfig
from ..enums import LinlinSet

logger = logging.getLogger(__name__)


class ConfigLinlin(BaseConfig):
    SETS = [member for member in LinlinSet]

    DEFAULT_FRAME_SIZE = (1536, 1024)
    CLOSED_NODES_FACTOR = 1.2  # You may need to sweep
    CLOSED_EDGES_FACTOR = .8  # You may need to sweep

    SYNTHETIC_GRAPH_NUMBER = 0  # Please set this yourself
    SYNTHETIC_NETWORK_NUMBER = 0  # Please set this yourself

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'linlin_input')

    # BASE_INPUT_PATH
    # ├── POSITION_DATA_DIR
    # ├── ADJ_MATRIX_DATA_DIR
    # └── IMAGES_DIR

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(set_name, _):
        positions = _load_positions(set_name)
        edge_list = _load_edge_list(set_name)
        from utils import build_graph
        original_network = build_graph(positions, edge_list, arg_type='edge_list')
        from config._utils import _transpose_network_pos
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(set_name, _):
        image = _load_raw_image(set_name)
        return image


def _load_positions(set_name):
    file_path = os.path.join(ConfigLinlin.BASE_INPUT_PATH, str(set_name), "nod_pos.csv")
    logger.info(f"Loading positions from {file_path}")
    return np.loadtxt(file_path, delimiter=',')


def _load_edge_list(set_name):
    # noinspection SpellCheckingInspection
    file_path = os.path.join(ConfigLinlin.BASE_INPUT_PATH, str(set_name), "edls.csv")
    logger.info(f"Loading edge list from {file_path}")
    return np.loadtxt(file_path, delimiter=',', dtype=int)[:, :2]


def _load_raw_image(set_name):
    if str(set_name) == "1-0":
        file_path = os.path.join(ConfigLinlin.BASE_INPUT_PATH, str(set_name), "1.tif")
    else:
        file_path = os.path.join(ConfigLinlin.BASE_INPUT_PATH, str(set_name), f"{str(set_name)}.tif")
    import cv2
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        err_msg = f"Failed to load image from file: {file_path}"
        logger.error(err_msg)
        return None
    return image
