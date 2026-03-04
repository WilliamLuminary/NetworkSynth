# src/configs/hybrid_mode/config_sample.py
"""
Hybrid mode sample configuration.

Phase 1: Generate HYBRID_ROWS × HYBRID_COLS seed tiles independently
          in parallel, each covering ~1×1 SYNTHETIC_FRAME_SIZE.  Quality-
          checked via multifractal error against the original network.

Phase 2: Assemble all seed tiles on a shared whiteboard (spacing = 2×
          SYNTHETIC_FRAME_SIZE per center) and continue BFS from their
          frontier nodes to fill the ~1×1 gaps and merge into a single
          connected network.

Result: a network roughly (2·HYBRID_ROWS) × (2·HYBRID_COLS) times
        larger than the original (in frame area).

Data layout
-----------
samples/hybrid_mode/
├── sample_{A,B,C,D}_pos.npy     (N×2 positions)
├── sample_{A,B,C,D}_mat.npy     (scipy sparse adjacency)
└── sample_{A,B,C,D}_image.tif   (grayscale background)
"""
import logging
import os
from typing import Tuple

import cv2
import numpy as np

from .._utils import (
    _resize_cv2_image,
    _transpose_network_pos,
    _trim_cv2_image,
)
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    """Hybrid mode sample — parallel seed tiles + frontier continuation."""

    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    # --- Hybrid layout parameters ---
    HYBRID_ROWS: int = 50
    HYBRID_COLS: int = 50
    PHASE2_MAX_ROUNDS: int = 500
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5

    # --- Network generation parameters ---
    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.5

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False

    # --- Input paths ---
    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "hybrid_mode",
    )

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        set_name = dataset_id[0]
        positions = _load_positions(set_name)
        mat = _load_sparse_matrix(set_name)

        from utils import build_graph

        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        image = _load_raw_image(dataset_id[0])
        if image is None:
            return None
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_pos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    logger.info("Loading positions from %s", path)
    return np.load(path, allow_pickle=True)


def _load_sparse_matrix(set_name: str):
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_mat.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency matrix file not found: {path}")
    logger.info("Loading adjacency matrix from %s", path)
    return np.load(path, allow_pickle=True).item()


def _load_raw_image(set_name: str):
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_image.tif")
    if not os.path.exists(path):
        logger.warning("No image at %s. Returning None.", path)
        return None
    logger.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", path)
    return image
