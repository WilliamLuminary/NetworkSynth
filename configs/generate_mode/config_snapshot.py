import logging
import os
from dataclasses import replace
from typing import Optional, Tuple

import cv2
import numpy as np

from graphs.synth_graph import SynthGraph
from utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


class SnapshotConfig(BaseConfig):
    MODE = "generate"

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

    RENDER_BFS_SNAPSHOT = replace(
        BaseConfig.RENDER_BFS_SNAPSHOT, node_size=0.5, line_width=0.5
    )
    RENDER_SYNTHETIC_GRAPH = replace(
        BaseConfig.RENDER_SYNTHETIC_GRAPH, node_size=0.5, line_width=0.5
    )

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples", "hybrid_mode")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId) -> SynthGraph:
        set_name = dataset_id[0]
        positions = _load_positions(set_name)
        mat = _load_sparse_matrix(set_name)

        from utils import build_graph

        original_network = build_graph(positions, mat)
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId) -> Optional[np.ndarray]:
        image = _load_raw_image(dataset_id[0])
        if image is None:
            return None
        image = trim_image(image)
        image = resize_image(image, SnapshotConfig.FRAME_SIZE)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(SnapshotConfig.BASE_INPUT_PATH, f"{set_name}_pos.npy")
    logger.info("Loading positions from %s", path)
    return np.load(path, allow_pickle=True)


def _load_sparse_matrix(set_name: str):
    path = os.path.join(SnapshotConfig.BASE_INPUT_PATH, f"{set_name}_mat.npy")
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
