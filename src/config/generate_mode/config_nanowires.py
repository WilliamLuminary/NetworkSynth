# src/config/generate_mode/config_nanowires.py
import logging
import os
from enum import Enum
from typing import Tuple

import cv2
import numpy as np

from utils import build_graph
from .._utils import _transpose_network_pos
from ..base_config import BaseConfig
from ..enums import SetName

logger = logging.getLogger(__name__)


# Generate NanowiresSet enum with members S1_0="1-0" through S1_224="1-224"
NanowiresSet = Enum(
    "NanowiresSet", {f"S1_{i}": f"1-{i}" for i in range(0, 225)}, type=SetName
)

_NANOWIRES_INPUT_DIR = "Nanowires-20250222-255-input"


class NanowiresConfig(BaseConfig):
    SETS = list(NanowiresSet)[:5]  # Use [:5] for testing, remove for full run

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
    FULL_ANALYSIS = False


    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, _NANOWIRES_INPUT_DIR)

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(set_name, _):
        """Load original network from position and edge list files."""
        positions = _load_positions(set_name)
        edge_list = _load_edge_list(set_name)
        original_network = build_graph(positions, edge_list, arg_type="edge_list")
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(set_name, _):
        """
        Load and process the background image.
        Add any image operations here (trim, resize, etc.) as needed.
        """
        directory = os.path.join(NanowiresConfig.BASE_INPUT_PATH, str(set_name))

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


def _load_positions(set_name) -> np.ndarray:
    """Load node positions from CSV file."""
    directory = os.path.join(NanowiresConfig.BASE_INPUT_PATH, str(set_name))
    file_path = os.path.join(directory, "nod_pos.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Positions file does not exist: {file_path}")

    logger.info(f"Loading positions from {file_path}")
    return np.loadtxt(file_path, delimiter=",")


def _load_edge_list(set_name) -> np.ndarray:
    """Load edge list from CSV file."""
    directory = os.path.join(NanowiresConfig.BASE_INPUT_PATH, str(set_name))
    file_path = os.path.join(directory, "edls.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Edge list file does not exist: {file_path}")

    logger.info(f"Loading edge list from {file_path}")
    return np.loadtxt(file_path, delimiter=",")
