import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

from utils import transpose_positions

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


def _generate_nanowires_datasets(count: int = 225) -> List[DatasetId]:
    return [DatasetId(f"1-{i}") for i in range(count)]


_NANOWIRES_INPUT_DIR = "Nanowires-20250222-255-input"


class ConfigNanowires(BaseConfig):
    MODE = "generate"

    DATASETS = _generate_nanowires_datasets()

    IMAGE_SIZE: Tuple[int, int] = (1024, 1536)
    FRAME_SIZE: Tuple[int, int] = (1536, 1024)
    SCALE_FACTOR: int = 3
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (
        FRAME_SIZE[0] * SCALE_FACTOR,
        FRAME_SIZE[1] * SCALE_FACTOR,
    )

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8
    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1
    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, _NANOWIRES_INPUT_DIR)

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        positions = _load_positions(dataset_id)
        edge_list = _load_edge_list(dataset_id)
        from utils import build_graph

        original_network = build_graph(positions, edge_list, arg_type="edge_list")
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        # For nanowires, dataset_id has single level: "1-0", "1-1", etc.
        directory = os.path.join(ConfigNanowires.BASE_INPUT_PATH, dataset_id[0])

        # Find any .tif file in the directory (naming varies: 1.tif, 1-1.tif, etc.)
        tif_files = [f for f in os.listdir(directory) if f.lower().endswith(".tif")]
        if not tif_files:
            logger.warning(f"No .tif image found in {directory}. Returning None.")
            return None

        file_path = os.path.join(directory, tif_files[0])
        logger.info(f"Loading image from {file_path}")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load image from {file_path}")
            return None

        return image


def _load_positions(dataset_id: DatasetId) -> np.ndarray:
    directory = os.path.join(ConfigNanowires.BASE_INPUT_PATH, dataset_id[0])
    file_path = os.path.join(directory, "nod_pos.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Positions file does not exist: {file_path}")

    logger.info(f"Loading positions from {file_path}")
    return np.loadtxt(file_path, delimiter=",")


def _load_edge_list(dataset_id: DatasetId) -> np.ndarray:
    directory = os.path.join(ConfigNanowires.BASE_INPUT_PATH, dataset_id[0])
    file_path = os.path.join(directory, "edls.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Edge list file does not exist: {file_path}")

    logger.info(f"Loading edge list from {file_path}")
    return np.loadtxt(file_path, delimiter=",")
