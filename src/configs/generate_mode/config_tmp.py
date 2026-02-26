# src/config/generate_mode/config_tmp.py
"""Temporary generation-mode config pointing at mosaic sample data."""
import logging
import os

import cv2
import numpy as np

from .._utils import _resize_cv2_image, _transpose_network_pos, _trim_cv2_image
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class ConfigTmp(BaseConfig):
    DATASETS = [DatasetId("sample_1")]

    IMAGE_SIZE = (2048, 2048)
    FRAME_SIZE = (510, 510)
    SYNTHETIC_FRAME_SIZE = FRAME_SIZE
    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.5

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15

    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False
    FULL_Q_BAND = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples", "mosaic_mode")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        set_name = dataset_id[0]
        path_pos = os.path.join(ConfigTmp.BASE_INPUT_PATH, f"{set_name}_pos.npy")
        path_mat = os.path.join(ConfigTmp.BASE_INPUT_PATH, f"{set_name}_mat.npy")

        if not os.path.exists(path_pos):
            raise FileNotFoundError(f"Positions file not found: {path_pos}")
        if not os.path.exists(path_mat):
            raise FileNotFoundError(f"Matrix file not found: {path_mat}")

        positions = np.load(path_pos, allow_pickle=True)
        mat = np.load(path_mat, allow_pickle=True).item()

        from utils import build_graph

        original_network = build_graph(positions, mat)
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
        set_name = dataset_id[0]
        path = os.path.join(ConfigTmp.BASE_INPUT_PATH, f"{set_name}_image.tif")
        if not os.path.exists(path):
            logger.warning("No image at %s. Returning None.", path)
            return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Failed to load image: %s", path)
            return None
        image = _trim_cv2_image(image)
        image = _resize_cv2_image(image)
        return image
