# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from networksynth.graphs.synth_graph import SynthGraph
from networksynth.utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId
from ..file_definitions import SYNTHETIC_DIR, SaveSpec, save_png, save_webp

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MODE = "mosaic"

    DATASETS = [DatasetId("sample_1")]

    GRID_ROWS: int = 2
    GRID_COLS: int = 2
    TILE_FRAME_SIZE: Tuple[int, int] = (510, 510)
    OVERLAP_MARGIN_FRACTION: float = 0.15

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

    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "mosaic_mode",
    )

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "png", save_png),
    )

    @staticmethod
    def load_original_network(dataset_id: DatasetId) -> SynthGraph:
        set_name = dataset_id[0]
        positions = _load_positions(set_name)
        mat = _load_sparse_matrix(set_name)

        from networksynth.utils import build_graph

        original_network = build_graph(positions, mat)
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId) -> Optional[np.ndarray]:
        image = _load_raw_image(dataset_id[0])
        if image is None:
            return None
        image = trim_image(image)
        image = resize_image(image, SampleConfig.FRAME_SIZE)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(
        SampleConfig.BASE_INPUT_PATH,
        f"{set_name}_pos.npy",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    logger.info("Loading positions from %s", path)
    return np.load(path, allow_pickle=True)


def _load_sparse_matrix(set_name: str):
    path = os.path.join(
        SampleConfig.BASE_INPUT_PATH,
        f"{set_name}_mat.npy",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency matrix file not found: {path}")
    logger.info("Loading adjacency matrix from %s", path)
    return np.load(path, allow_pickle=True).item()


def _load_raw_image(set_name: str):
    path = os.path.join(
        SampleConfig.BASE_INPUT_PATH,
        f"{set_name}_image.tif",
    )
    if not os.path.exists(path):
        logger.warning("No image at %s. Returning None.", path)
        return None

    logger.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", path)
    return image
