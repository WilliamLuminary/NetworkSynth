# src/configs/generate_mode/config_snapshot.py
"""
Generate-mode config for sample_A.

Uses the hybrid-mode sample_A data with its tuned node/edge factors.
Base class for Snapshot1x1Config / Snapshot3x3Config variants.
"""
import logging
import os
from typing import Tuple

import cv2
import numpy as np

from .._utils import _resize_cv2_image, _transpose_network_pos, _trim_cv2_image
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class SnapshotConfig(BaseConfig):
    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1530, 1530)

    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.3

    SNAPSHOT_INTERVAL = 0
    SELECT_BEST = 0

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 50
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False

    # Shared visual style for both BFS snapshots and final synthetic plots.
    # Adjust node_size / line_width when SYNTHETIC_FRAME_SIZE differs
    # from FRAME_SIZE — larger frames need thinner strokes.
    #   1×1 (510):  node_size=6.0, line_width=3.0
    #   3×3 (1530): node_size=2.0, line_width=1.0
    PLOT_STYLE: dict = {
        "dpi": 300,
        "node_size": 0.5,
        "line_width": 0.5,
    }

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples", "hybrid_mode")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

        from configs.file_definitions import FILE_CONFIGURATIONS

        style = cls.PLOT_STYLE
        syn_cfg = FILE_CONFIGURATIONS["synthetic_graph"]
        syn_cfg.node_size = style.get("node_size", syn_cfg.node_size)
        syn_cfg.line_width = style.get("line_width", syn_cfg.line_width)

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
        image = _resize_cv2_image(image, SnapshotConfig.FRAME_SIZE)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(SnapshotConfig.BASE_INPUT_PATH, f"{set_name}_pos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    logger.info("Loading positions from %s", path)
    return np.load(path, allow_pickle=True)


def _load_sparse_matrix(set_name: str):
    path = os.path.join(SnapshotConfig.BASE_INPUT_PATH, f"{set_name}_mat.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency matrix file not found: {path}")
    logger.info("Loading adjacency matrix from %s", path)
    return np.load(path, allow_pickle=True).item()


def _load_raw_image(set_name: str):
    path = os.path.join(SnapshotConfig.BASE_INPUT_PATH, f"{set_name}_image.tif")
    if not os.path.exists(path):
        logger.warning("No image at %s. Returning None.", path)
        return None
    logger.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", path)
    return image
