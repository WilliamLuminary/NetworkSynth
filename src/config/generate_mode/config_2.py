# src/config/generate_mode/config_2.py
import logging
import os
import re

import cv2
import numpy as np

from utils import build_graph
from .._utils import _find_file_with_pattern, _resize_cv2_image, _transpose_network_pos, _trim_cv2_image
from ..base_config import BaseConfig
from ..enums import NewSet

logger = logging.getLogger(__name__)


class Config2(BaseConfig):
    SETS = [NewSet.S4]

    TRIM_SIZE = (0, 116, 0, 0)  # (Top, Bottom, Left, Right)
    DEFAULT_FRAME_SIZE = (1887 // 4, 2048 // 4)

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    MEASURE_WEIGHTED = False
    FULL_ANALYSIS = False
    ERROR_TOLERANCE = .3

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'new_input')
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
    directory_path = os.path.join(Config2.POSITION_DATA_DIR, resolution)
    # noinspection SpellCheckingInspection
    file_path = _find_file_with_pattern(directory_path,
                                        rf"W-\d+-\d+-\d+_{re.escape(str(set_name))}_postion\.npy",
                                        f'positions_of_nodes for {set_name}')
    logger.info(f"Positions file loaded: {file_path}")
    positions = np.load(file_path, allow_pickle=True)
    return positions


def _load_sparse_matrix(set_name, resolution):
    directory_path = os.path.join(Config2.ADJ_MATRIX_DATA_DIR, resolution)
    file_path = _find_file_with_pattern(directory_path, r"sparse_matrices\.npz", details="sparse matrix")
    matrix_data = np.load(file_path, allow_pickle=True)

    key_pattern = rf"W-\d+-\d+-\d+_{re.escape(str(set_name))}_EL"
    matching_keys = [
        key for key in matrix_data.keys()
        if re.fullmatch(key_pattern, key)
    ]

    if len(matching_keys) == 1:
        logger.info(f"Sparse matrix loaded: {matching_keys[0]}")
        return matrix_data[matching_keys[0]].item()
    elif len(matching_keys) > 1:
        raise FileExistsError(f"Multiple matching sparse matrices found: {matching_keys}")
    else:
        available_keys = list(matrix_data.keys())
        raise KeyError(
            f"No matching sparse matrix found for set '{set_name}'. "
            f"Available keys: {available_keys}"
        )


def _load_raw_image(set_name, resolution):
    directory_path = os.path.join(Config2.IMAGES_DIR, resolution)

    pattern = re.compile(
        rf"W-\d+-\d+-\d+_{re.escape(str(set_name))}\.(tif|png|jpg)",
        re.IGNORECASE
    )

    file_path = _find_file_with_pattern(directory_path, pattern, f'image for {set_name}')
    if file_path is None:
        logger.warning(f"Background image is None.")
        return

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    logger.info(f"Image file loaded: {file_path}")
    if image is None:
        err_msg = f"Failed to load image from file: {file_path}"
        logger.error(err_msg)
        return None

    return image
