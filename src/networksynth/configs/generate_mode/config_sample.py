# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from networksynth.graphs.synth_graph import SynthGraph
from networksynth.utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


def _generate_sample_datasets() -> List[DatasetId]:
    return [DatasetId("sample_1"), DatasetId("sample_2"), DatasetId("sample_3")]


class SampleConfig(BaseConfig):
    MODE = "generate"

    DATASETS = _generate_sample_datasets()

    IMAGE_SIZE = (1887, 2048)
    FRAME_SIZE = (1887 // 4, 2048 // 4)
    SYNTHETIC_FRAME_SIZE = FRAME_SIZE
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH, "samples", "generate_mode"
    )
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, "")
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, "")
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, "")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId) -> SynthGraph:
        positions = _load_positions(cls.POSITION_DATA_DIR, dataset_id)
        mat = _load_sparse_matrix(cls.ADJ_MATRIX_DATA_DIR, dataset_id)
        from networksynth.utils import build_graph

        original_network = build_graph(positions, mat)
        transpose_positions(original_network)
        return original_network

    @classmethod
    def load_original_image(cls, dataset_id: DatasetId) -> Optional[np.ndarray]:
        image = _load_raw_image(cls.IMAGES_DIR, dataset_id)
        if image is None:
            return None
        image = trim_image(image)
        image = resize_image(image, cls.FRAME_SIZE)
        return image


def _load_positions(directory: str, dataset_id: DatasetId) -> np.ndarray:
    file_path = os.path.join(directory, f"{dataset_id[0]}_pos.npy")

    logger.info(f"Loading positions from {file_path}")
    return np.load(file_path, allow_pickle=True)


def _load_sparse_matrix(directory: str, dataset_id: DatasetId):
    file_path = os.path.join(directory, f"{dataset_id[0]}_mat.npy")

    logger.info(f"Loading adjacency matrix from {file_path}")
    return np.load(file_path, allow_pickle=True).item()


def _load_raw_image(directory: str, dataset_id: DatasetId):
    file_path = os.path.join(directory, f"{dataset_id[0]}_image.tif")

    if not os.path.exists(file_path):
        logger.warning(f"No image found at {file_path}. Returning None.")
        return None

    logger.info(f"Loading image from {file_path}")
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning(f"Failed to load image from {file_path}")
    return image
