import logging
import os
from typing import Optional

import cv2
import numpy as np

from graphs.synth_graph import SynthGraph
from utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


class ConfigTmp(BaseConfig):
    MODE = "generate"

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
    FULL_Q_BAND = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples", "mosaic_mode")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId) -> SynthGraph:
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
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId) -> Optional[np.ndarray]:
        set_name = dataset_id[0]
        path = os.path.join(ConfigTmp.BASE_INPUT_PATH, f"{set_name}_image.tif")
        if not os.path.exists(path):
            logger.warning("No image at %s. Returning None.", path)
            return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Failed to load image: %s", path)
            return None
        image = trim_image(image)
        image = resize_image(image, ConfigTmp.FRAME_SIZE)
        return image
