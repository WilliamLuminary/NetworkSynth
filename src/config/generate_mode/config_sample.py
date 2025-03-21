# src/config/generate_mode/config_sample.py
import logging
import os

import cv2
import numpy as np

from .._utils import _resize_cv2_image, _transpose_network_pos, _trim_cv2_image
from ..base_config import BaseConfig
from ..enums import IdleResolution, SampleSet

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    SETS = [SampleSet.Sample1, SampleSet.Sample2, SampleSet.Sample3]

    DEFAULT_FRAME_SIZE = (1887 // 4, 2048 // 4)
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = .8

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'sample_input', 'generate_mode')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, '')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, '')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, '')

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
        mat = _load_sparse_matrix(set_name)
        from utils import build_graph
        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(set_name, _):
        image = _load_raw_image(set_name)
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image)
        return image


def _load_positions(set_name):
    file_path = os.path.join(SampleConfig.POSITION_DATA_DIR, f"{str(set_name)}_pos.npy")

    if not os.path.exists(file_path):
        msg = f"Positions file does not exist: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading positions from {file_path}")
    return np.load(file_path, allow_pickle=True)


def _load_sparse_matrix(set_name):
    file_path = os.path.join(SampleConfig.ADJ_MATRIX_DATA_DIR, f"{str(set_name)}_mat.npy")

    if not os.path.exists(file_path):
        msg = f"Sparse matrix file does not exist: {file_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading adjacency matrix from {file_path}")
    return np.load(file_path, allow_pickle=True).item()


def _load_raw_image(set_name):
    file_path = os.path.join(SampleConfig.IMAGES_DIR, f"{str(set_name)}_image.tif")

    if not os.path.exists(file_path):
        logger.warning(f"No image found at {file_path}. Returning None.")
        return None

    logger.info(f"Loading image from {file_path}")
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning(f"Failed to load image from {file_path}")
    return image
