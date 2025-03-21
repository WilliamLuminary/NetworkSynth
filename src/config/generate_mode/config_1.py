# src/config/generate_mode/config_1.py
import logging
import os
import re

import cv2
import numpy as np

from utils import build_graph
from .. import SetName
from .._utils import _find_file_with_pattern, _resize_cv2_image, _transpose_network_pos, _trim_cv2_image
from ..base_config import BaseConfig
from ..enums import LinlinSet, OldResolution, OldSet

logger = logging.getLogger(__name__)


class Config1(BaseConfig):
    SETS = [OldSet.A]
    RESOLUTIONS = [OldResolution.X10K]

    DEFAULT_FRAME_SIZE = (510, 510)
    CLOSED_NODES_FACTOR = 1.5
    CLOSED_EDGES_FACTOR = 1
    # For the 10_kx image, node fac should be 1.5, and edge fac should be 1

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'old_input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(set_name, resolution):
        positions = _load_positions(set_name, resolution)
        mat = _load_sparse_matrix(set_name, resolution)
        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(set_name, resolution):
        image = _load_raw_image(set_name, resolution)
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image)
        return image


def _load_positions(set_name, resolution):
    directory_path = Config1.POSITION_DATA_DIR
    file_path = _find_file_with_pattern(directory_path,
                                        rf"{re.escape(set_name)}_{re.escape(resolution)}.*\.npy",
                                        f'positions_of_nodes for {set_name}')
    logger.info(f"Positions file loaded: {file_path}")
    positions = np.load(file_path, allow_pickle=True)
    return positions


def _load_sparse_matrix(set_name, resolution):
    directory_path = Config1.ADJ_MATRIX_DATA_DIR
    file_path = _find_file_with_pattern(directory_path,
                                        rf"sparse_matrices_{re.escape(resolution)}.*\.npz",
                                        'sparse matrix')
    matrix_data = np.load(file_path, allow_pickle=True)

    set_name = set_name
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


def _load_raw_image(set_name, resolution):
    directory_path = os.path.join(Config1.IMAGES_DIR, set_name)
    if set_name == 'B':
        directory_path = os.path.join(directory_path, '1811')
        logger.warning("For set B, using the 1811 subdirectory for images.")

    file_path = _find_file_with_pattern(directory_path,
                                        rf"\b{re.escape(resolution)}\b.*\.(tif|png|jpg)"
                                        , f'image for {set_name}')
    if file_path is None:
        logger.warning(f"Background image is None.")
        return None

    logger.info(f"Image file loaded: {file_path}")

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        err_msg = f"Failed to load image from file: {file_path}"
        logger.error(err_msg)
        return None
    return image
