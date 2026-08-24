import logging
import os
import re
from typing import Tuple

import cv2
import numpy as np

from .._utils import (
    _find_file_with_pattern,
    _resize_cv2_image,
    _transpose_network_pos,
    _trim_cv2_image,
)
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class Config1(BaseConfig):

    MODE = "mosaic"

    # --- Dataset ---
    DATASETS = [DatasetId("A", "20kX")]

    # --- Mosaic grid parameters ---
    GRID_ROWS: int = 100
    GRID_COLS: int = 100
    TILE_FRAME_SIZE: Tuple[int, int] = (510, 510)

    # Per-side overlap (px) = fraction × tile_frame_size[dim]
    OVERLAP_MARGIN_FRACTION: float = 0.15

    # --- Network generation parameters (per tile) ---
    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = TILE_FRAME_SIZE

    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.5

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True

    # --- Input paths (old_input format) ---
    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "old_input")
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, "position")
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, "sparse_matrices")
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, "Original Graphs")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def save_synthetic_graph(cls):
        from ..file_definitions import save_png, save_webp

        return [
            ("synthetic", "synthetic_graph", "webp", save_webp),
            ("synthetic", "synthetic_graph", "png", save_png),
        ]

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        positions = _load_positions(dataset_id)
        mat = _load_sparse_matrix(dataset_id)
        from utils import build_graph

        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        image = _load_raw_image(dataset_id)
        if image is None:
            return None
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image, Config1.FRAME_SIZE)
        return image


# ------------------------------------------------------------------ #
# Data loaders (old_input format, two-level DatasetId)
# ------------------------------------------------------------------ #


def _load_positions(dataset_id: DatasetId) -> np.ndarray:
    set_name, resolution = dataset_id[0], dataset_id[1]
    file_path = _find_file_with_pattern(
        Config1.POSITION_DATA_DIR,
        rf"{re.escape(set_name)}_{re.escape(resolution)}" rf".*\.npy",
        f"positions for {set_name} {resolution}",
    )
    logger.info("Loading positions from %s", file_path)
    return np.load(file_path, allow_pickle=True)


def _load_sparse_matrix(dataset_id: DatasetId):
    set_name, resolution = dataset_id[0], dataset_id[1]
    file_path = _find_file_with_pattern(
        Config1.ADJ_MATRIX_DATA_DIR,
        rf"sparse_matrices_{re.escape(resolution)}" rf".*\.npz",
        "sparse matrix",
    )
    matrix_data = np.load(file_path, allow_pickle=True)

    matrix_key = set_name
    if set_name == "C":
        if "C1" in matrix_data:
            matrix_key = "C1"
        elif "C" not in matrix_data:
            available = list(matrix_data.keys())
            raise KeyError(
                f"Neither 'C' nor 'C1' in sparse matrix. " f"Available: {available}"
            )

    if matrix_key not in matrix_data:
        available = list(matrix_data.keys())
        raise KeyError(f"Set '{matrix_key}' not found. " f"Available: {available}")

    return matrix_data[matrix_key].item()


def _load_raw_image(dataset_id: DatasetId):
    set_name, resolution = dataset_id[0], dataset_id[1]
    directory_path = os.path.join(Config1.IMAGES_DIR, set_name)

    if set_name == "B":
        directory_path = os.path.join(directory_path, "1811")
        logger.warning("For set B, using 1811 subdirectory.")

    file_path = _find_file_with_pattern(
        directory_path,
        rf"\b{re.escape(resolution)}\b" rf".*\.(tif|png|jpg)",
        f"image for {set_name} {resolution}",
    )
    if file_path is None:
        logger.warning("Background image is None.")
        return None

    logger.info("Loading image from %s", file_path)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", file_path)
    return image
