import csv
import logging
import os
from dataclasses import replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from graphs.synth_graph import SynthGraph
from utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


class ConfigDickson(BaseConfig):
    MODE = "hybrid"

    LOG_MEMORY = True

    DATASETS = [
        DatasetId("gel1"),
        DatasetId("gel2"),
        DatasetId("gel3"),
        DatasetId("gel4"),
    ]

    TARGET_SCALE: Tuple[int, int] = (40, 40)

    NUM_CENTERS: int = 500
    PHASE2_MAX_ROUNDS: int = 500
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5

    TILE_FRAME_SIZE: Tuple[int, int] = None
    TILE_FRAME_FACTOR: float = 0.5

    IMAGE_SIZE: Tuple[int, int] = (730, 1030)
    FRAME_SIZE: Tuple[int, int] = (1030, 730)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1030, 730)

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 1.5

    DATASET_FACTORS: Dict[str, Tuple[float, float]] = {}

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 5

    RENDER_ORIGINAL_GRAPH = replace(BaseConfig.RENDER_ORIGINAL_GRAPH, node_size=3.0)

    RENDER_HYBRID_GRAPH = replace(BaseConfig.RENDER_HYBRID_GRAPH, max_px=8000)
    RENDER_HYBRID_SNAPSHOT = replace(BaseConfig.RENDER_HYBRID_SNAPSHOT, max_px=8000)

    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "dickson")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId) -> SynthGraph:
        set_name = dataset_id[0]
        positions = _load_positions(set_name)
        mat = _load_weighted_matrix(set_name, len(positions))

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
        image = resize_image(image, ConfigDickson.FRAME_SIZE)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(ConfigDickson.BASE_INPUT_PATH, f"{set_name}_NodePositions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    logger.info("Loading positions from %s", path)
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def _load_weighted_matrix(set_name: str, n: int):
    import scipy.sparse as sp

    path = os.path.join(ConfigDickson.BASE_INPUT_PATH, f"{set_name}_EdgeList.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Edge list file not found: {path}")
    logger.info("Loading weighted edge list from %s", path)

    rows, cols, weights = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for record in reader:
            u, v, w = int(record[0]), int(record[1]), float(record[2])
            rows.append(u)
            cols.append(v)
            weights.append(w)

    r = np.array(rows + cols)
    c = np.array(cols + rows)
    data = np.array(weights + weights, dtype=np.float64)
    return sp.coo_matrix((data, (r, c)), shape=(n, n))


def _load_raw_image(set_name: str):
    path = os.path.join(ConfigDickson.BASE_INPUT_PATH, f"{set_name}.bmp")
    if not os.path.exists(path):
        logger.warning("No image at %s. Returning None.", path)
        return None
    logger.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", path)
    return image
