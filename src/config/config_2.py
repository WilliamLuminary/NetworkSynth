# src/config/config_2.py
import logging
import os
import re

import cv2
import numpy as np

from .config import Config
from .enums import Resolution, SetName

logger = logging.getLogger(__name__)


class Config2(Config):
    SETS = [SetName.S4, SetName.S8, SetName.S11, SetName.S14, SetName.S17, SetName.S20, SetName.S23, SetName.S26,
            SetName.S29, SetName.S32]
    RESOLUTIONS = [Resolution.NA]

    DEFAULT_FRAME_SIZE = (470, 470)
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    # For 10_kx image, node fac should be 1.5, and edge fac should be 1

    SYNTHETIC_GRAPH_NUMBER = 10
    SYNTHETIC_NETWORK_NUMBER = 300

    MEASURE_WEIGHTED = False
    ERROR_TOLERANCE = .3

    BASE_INPUT_PATH = os.path.join(Config.BASE_INPUT_PATH, 'new_input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')

    @classmethod
    def initialize(cls):
        cls._update_attrs_in_base_config()
        super()._setup_logger(details="new")
        Config.POSITION_DATA_FUNC = cls._load_positions
        Config.ADJ_MATRIX_DATA_FUNC = cls._load_sparse_matrix
        Config.IMAGES_FUNC = cls._load_image

    @staticmethod
    def _load_positions(set_name, resolution):
        directory_path = os.path.join(Config2.POSITION_DATA_DIR, resolution)
        pattern = re.compile(
            rf"W-\d+-\d+-\d+_{re.escape(str(set_name))}_postion\.npy",
            re.IGNORECASE
        )  # Only difference
        file_path = Config._find_file_with_pattern(
            directory_path, pattern, f'positions_of_nodes for {set_name}'
        )
        logger.info(f"Positions file loaded: {file_path}")
        positions = np.load(file_path, allow_pickle=True)
        return positions

    @staticmethod
    def _load_sparse_matrix(set_name, resolution):
        directory_path = os.path.join(Config2.ADJ_MATRIX_DATA_DIR, resolution)
        pattern = re.compile(r"sparse_matrices\.npz", re.IGNORECASE)

        file_path = Config._find_file_with_pattern(
            directory_path, pattern, details="sparse matrix"
        )
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

    @staticmethod
    def _load_image(set_name, resolution):
        directory_path = os.path.join(Config2.IMAGES_DIR, resolution)

        pattern = re.compile(
            rf"W-\d+-\d+-\d+_{re.escape(str(set_name))}\.(tif|png|jpg)",
            re.IGNORECASE
        )

        file_path = Config._find_file_with_pattern(
            directory_path, pattern, f'image for {set_name}'
        )
        if file_path is None:
            logger.warning(f"Background image is None.")
            return

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        logger.info(f"Image file loaded: {file_path}")
        if image is None:
            err_msg = f"Failed to load image from file: {file_path}"
            logger.warning(err_msg)
            return None

        return image
