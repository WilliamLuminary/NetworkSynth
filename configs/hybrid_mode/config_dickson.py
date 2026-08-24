import csv
import logging
import os
from typing import Dict, Tuple

import cv2
import numpy as np

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

    # --- Hybrid layout parameters ---
    # 40x40 tiling of the original frame.
    TARGET_SCALE: Tuple[int, int] = (40, 40)

    NUM_CENTERS: int = 500
    PHASE2_MAX_ROUNDS: int = 500
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5

    # --- Phase 1 tile frame sizing (auto) ---
    TILE_FRAME_SIZE: Tuple[int, int] = None
    TILE_FRAME_FACTOR: float = 0.5
    MIN_TILE_FRAME: float = 382.0

    # --- Network generation parameters ---
    # Positions are transposed to landscape (see load_original_network) so the
    # network aspect matches the landscape .bmp backgrounds. After transpose
    # the extent is x up to ~1022, y up to ~722.
    #   FRAME_SIZE / SYNTHETIC_FRAME_SIZE = (x_range, y_range)
    #   IMAGE_SIZE = (height, width)
    IMAGE_SIZE: Tuple[int, int] = (730, 1030)
    FRAME_SIZE: Tuple[int, int] = (1030, 730)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1030, 730)

    # The gel networks are ~2x larger in extent but have ~half the edge
    # length of the sample data, so the sample's edge factor (2.0) chokes
    # Phase 2 frontier growth and tiles fail to stitch (40x40 -> hundreds of
    # disconnected components). CLOSED_EDGES_FACTOR=1.5 unchokes growth and
    # the whiteboard connects into a single component (verified on gel1).
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 1.5

    # Per-dataset overrides for (CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR).
    # No sweep has been run for the Dickson data yet; all use the defaults.
    DATASET_FACTORS: Dict[str, Tuple[float, float]] = {}

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    # Lower edge factor makes each tile denser/slower to generate, so cap
    # retries: tiles that pass tolerance early exit immediately; the cap
    # bounds the worst case for tiles that never pass.
    MAX_ATTEMPTS = 5

    # Shrink the node markers in the original_graph render (dense gel
    # networks look cleaner with smaller dots). Default elsewhere is 1.0.
    ORIGINAL_GRAPH_NODE_SCALE = 0.5

    # Cap the synthetic-graph render size. At the 16383px WebP limit an 11M+
    # node network produces a ~190-megapixel image that most viewers refuse
    # to open; 8000px stays detailed but opens everywhere.
    RENDER_MAX_PX = 8000
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True

    # --- Input paths ---
    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "dickson")

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId):
        set_name = dataset_id[0]
        positions = _load_positions(set_name)
        mat = _load_weighted_matrix(set_name, len(positions))

        from utils import build_graph

        original_network = build_graph(positions, mat)
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId):
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
        next(reader)  # header: Source,Target,Weight,...
        for record in reader:
            u, v, w = int(record[0]), int(record[1]), float(record[2])
            rows.append(u)
            cols.append(v)
            weights.append(w)

    # Symmetrize: the edge list stores each undirected edge once.
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
