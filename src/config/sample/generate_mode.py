# src/config/sample/generate_mode.py
import logging
import os

import cv2
import numpy as np

from config.base_config import BaseConfig
from config.enums import Resolution, SetName

logger = logging.getLogger(__name__)


class GenerateModeConfigSample(BaseConfig):
    SETS = [SetName.Sample1, SetName.Sample2, SetName.Sample3]
    RESOLUTIONS = [Resolution.NA]

    DEFAULT_FRAME_SIZE = (1887 // 4, 2048 // 4)
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = .8
    # For 10_kx image, node fac should be 1.5, and edge fac should be 1

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
        cls.POSITION_DATA_FUNC = cls._load_positions
        cls.ADJ_MATRIX_DATA_FUNC = cls._load_sparse_matrix
        cls.IMAGES_FUNC = cls._load_image
        cls._update_attrs_in_base_config()

    @staticmethod
    def _load_positions(set_name, _):
        directory_path = GenerateModeConfigSample.POSITION_DATA_DIR
        file_name = f"{str(set_name)}_pos.npy"
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            msg = f"Positions file does not exist: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading positions from {file_path}")
        return np.load(file_path, allow_pickle=True)

    @staticmethod
    def _load_sparse_matrix(set_name, _):
        directory_path = GenerateModeConfigSample.ADJ_MATRIX_DATA_DIR
        file_name = f"{str(set_name)}_mat.npy"
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            msg = f"Sparse matrix file does not exist: {file_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading adjacency matrix from {file_path}")
        return np.load(file_path, allow_pickle=True).item()

    @staticmethod
    def _load_image(set_name, _):
        directory_path = GenerateModeConfigSample.IMAGES_DIR
        file_name = f"{str(set_name)}_image.tif"
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            logger.warning(f"No image found at {file_path}. Returning None.")
            return None

        logger.info(f"Loading image from {file_path}")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load image from {file_path}")
        return image
