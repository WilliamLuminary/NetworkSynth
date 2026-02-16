# src/config/mosaic_mode/config_sample.py
"""
Mosaic mode sample configuration.

Generates a GRID_ROWS x GRID_COLS grid of small tile networks,
then stitches them into one large network by merging close nodes
in the overlap regions between adjacent tiles.

Uses the *old_input* dataset format (A 20kX by default):
  - positions  : .npy  from  old_input/position/
  - adjacency  : .npz  from  old_input/sparse_matrices/
  - images     : .tif  from  old_input/Original Graphs/<set>/
"""
import logging
import os
import re
from typing import Tuple

import cv2
import numpy as np

from utils import build_graph

from .._utils import (
    _find_file_with_pattern,
    _resize_cv2_image,
    _transpose_network_pos,
    _trim_cv2_image,
)
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    """
    Mosaic mode: generate a grid of tile networks and stitch them together.

    Mosaic-specific parameters:
        GRID_ROWS / GRID_COLS: Number of tiles in each dimension.
        TILE_FRAME_SIZE: Frame size (X, Y) of each tile.
        OVERLAP_MARGIN_FRACTION: Fraction of tile frame used
            as overlap per tile side.
        NUM_WORKERS: Number of parallel processes for tile generation.
    """

    # --- Dataset ---
    DATASETS = [DatasetId("A", "20kX")]

    # --- Mosaic Grid Parameters ---
    GRID_ROWS: int = 100
    GRID_COLS: int = 100
    TILE_FRAME_SIZE: Tuple[int, int] = (510, 510)

    # Overlap margin as a fraction of the tile frame size.
    # Per-side overlap (px) =
    #   OVERLAP_MARGIN_FRACTION * TILE_FRAME_SIZE[dim]
    # 0.15 × 510 ≈ 76 px per side (~6 avg edge lengths for A 20kX).
    OVERLAP_MARGIN_FRACTION: float = 0.15

    # --- Network Generation Parameters (identical for all tiles) ---
    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = TILE_FRAME_SIZE

    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.5

    # Not used in mosaic mode (tiles, not individual synths)
    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False

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
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        """Load original network from .npy positions + .npz sparse matrix."""
        positions = _load_positions(dataset_id)
        mat = _load_sparse_matrix(dataset_id)
        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        """Load and process the original TIF image."""
        image = _load_raw_image(dataset_id)
        if image is None:
            return None
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image)
        return image


# ------------------------------------------------------------------ #
# Data loaders  (mirrors generate_mode/config_1.py for old_input)
# ------------------------------------------------------------------ #
def _load_positions(dataset_id: DatasetId) -> np.ndarray:
    """Load node positions from .npy file.

    dataset_id[0] = set_name (A), dataset_id[1] = resolution (20kX).
    """
    set_name, resolution = dataset_id[0], dataset_id[1]
    file_path = _find_file_with_pattern(
        SampleConfig.POSITION_DATA_DIR,
        rf"{re.escape(set_name)}_{re.escape(resolution)}.*\.npy",
        f"positions for {set_name} {resolution}",
    )
    logger.info("Loading positions from %s", file_path)
    return np.load(file_path, allow_pickle=True)


def _load_sparse_matrix(dataset_id: DatasetId):
    """Load adjacency sparse matrix from .npz file.

    dataset_id[0] = set_name (A), dataset_id[1] = resolution (20kX).
    """
    set_name, resolution = dataset_id[0], dataset_id[1]
    file_path = _find_file_with_pattern(
        SampleConfig.ADJ_MATRIX_DATA_DIR,
        rf"sparse_matrices_{re.escape(resolution)}.*\.npz",
        "sparse matrix",
    )
    matrix_data = np.load(file_path, allow_pickle=True)

    matrix_key = set_name
    if set_name == "C":
        if "C1" in matrix_data:
            matrix_key = "C1"
        elif "C" in matrix_data:
            matrix_key = "C"
        else:
            available = list(matrix_data.keys())
            raise KeyError(
                f"Neither 'C' nor 'C1' found in sparse matrix. "
                f"Available: {available}"
            )

    if matrix_key not in matrix_data:
        available = list(matrix_data.keys())
        raise KeyError(
            f"Set '{matrix_key}' not found in sparse matrix. " f"Available: {available}"
        )

    return matrix_data[matrix_key].item()


def _load_raw_image(dataset_id: DatasetId):
    """Load raw TIF image for the dataset.

    dataset_id[0] = set_name (A), dataset_id[1] = resolution (20kX).
    """
    set_name, resolution = dataset_id[0], dataset_id[1]
    directory_path = os.path.join(SampleConfig.IMAGES_DIR, set_name)

    if set_name == "B":
        directory_path = os.path.join(directory_path, "1811")
        logger.warning("For set B, using the 1811 subdirectory for images.")

    file_path = _find_file_with_pattern(
        directory_path,
        rf"\b{re.escape(resolution)}\b.*\.(tif|png|jpg)",
        f"image for {set_name} {resolution}",
    )
    if file_path is None:
        logger.warning("Background image is None.")
        return None

    logger.info("Loading image from %s", file_path)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image from %s", file_path)
    return image
