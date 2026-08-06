# src/configs/hybrid_mode/config_sample.py
"""
Hybrid mode sample configuration.

Phase 1: Generate seed tiles independently in parallel, each covering
          ~1×1 SYNTHETIC_FRAME_SIZE.  Quality-checked via multifractal
          error against the original network.

Phase 2: Assemble all seed tiles on a shared whiteboard and continue
          BFS from frontier nodes to fill gaps and merge into a single
          connected network.

Result: a network whose whiteboard spans TARGET_SCALE times the
        original frame in each dimension.

Data layout
-----------
samples/hybrid_mode/
├── sample_{A,B,C,D}_pos.npy     (N×2 positions)
├── sample_{A,B,C,D}_mat.npy     (scipy sparse adjacency)
└── sample_{A,B,C,D}_image.tif   (grayscale background)
"""
import logging
import os
from typing import Dict, Tuple

import cv2
import numpy as np

from .._utils import (
    _resize_cv2_image,
    _transpose_network_pos,
    _trim_cv2_image,
)
from ..base_config import BaseConfig
from ..enums import DatasetId

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    """Hybrid mode sample — parallel seed tiles + frontier continuation."""

    LOG_MEMORY = True

    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    # --- Hybrid layout parameters ---

    # Desired output scale relative to the original frame, per axis.
    # E.g. (100, 100) means the whiteboard is 100× the original frame
    # width and 100× the original frame height.
    TARGET_SCALE: Tuple[int, int] = (100, 100)

    NUM_CENTERS: int = 2000
    PHASE2_MAX_ROUNDS: int = 500
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5

    # --- Phase 1 tile frame sizing ---
    # How big each seed tile is allowed to grow before Phase 2 takes over.
    #
    # Fixed mode: set TILE_FRAME_SIZE = (w, h) to give every tile the same
    # frame. Leave it None to size each tile automatically.
    #
    # Auto mode (TILE_FRAME_SIZE = None): each tile's frame side =
    # nearest-neighbor distance * TILE_FRAME_FACTOR, clamped up to
    # MIN_TILE_FRAME so tiles never get too small to produce a usable
    # network. Factor 0.5 leaves a gap (~half the spacing) for Phase 2 to
    # stitch; smaller factor = wider gap.
    TILE_FRAME_SIZE: Tuple[int, int] = None
    TILE_FRAME_FACTOR: float = 0.5
    MIN_TILE_FRAME: float = 382.0

    # --- Network generation parameters ---
    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    CLOSED_NODES_FACTOR = 1.4
    CLOSED_EDGES_FACTOR = 2.0

    # Per-dataset overrides for (CLOSED_NODES_FACTOR, CLOSED_EDGES_FACTOR).
    # Datasets not listed here use the defaults above.
    DATASET_FACTORS: Dict[str, Tuple[float, float]] = {
        # "sample_A": (1.0, 1.5),
        # "sample_B": (1.8, 1.5),
        # "sample_C": (1.3, 1.6),
        # "sample_D": (1.5, 1.3),
        # These are the values from the sweep
        "sample_A": (1.0, 1.4),
        "sample_B": (1.2, 0.9),
        "sample_C": (1.0, 1.2),
        "sample_D": (1.2, 0.9),
    }

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 50
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True
    FULL_ANALYSIS = False

    # --- Input paths ---
    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "hybrid_mode",
    )

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @classmethod
    def save_synthetic_graph(cls):
        from ..file_definitions import save_png, save_webp

        return [
            ("synthetic", "synthetic_graph", "webp", save_webp),
            ("synthetic", "synthetic_graph", "png", save_png),
        ]

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
        image = _resize_cv2_image(image)
        return image


def _load_positions(set_name: str) -> np.ndarray:
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_pos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Positions file not found: {path}")
    logger.info("Loading positions from %s", path)
    return np.load(path, allow_pickle=True)


def _load_sparse_matrix(set_name: str):
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_mat.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency matrix file not found: {path}")
    logger.info("Loading adjacency matrix from %s", path)
    return np.load(path, allow_pickle=True).item()


# ------------------------------------------------------------------ #
# CSV input: swap the two loaders above for these when the network is
# stored as a node-position table plus an edge list instead of .npy.
# Expected headers: "<set>_NodePositions.csv" -> x,y
#                   "<set>_EdgeList.csv"      -> Source,Target,Weight
# Requires `import csv` at the top of this file.
# ------------------------------------------------------------------ #
# def _load_positions(set_name: str) -> np.ndarray:
#     path = os.path.join(
#         SampleConfig.BASE_INPUT_PATH, f"{set_name}_NodePositions.csv"
#     )
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Positions file not found: {path}")
#     logger.info("Loading positions from %s", path)
#     return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
#
#
# def _load_sparse_matrix(set_name: str, n: int):
#     import scipy.sparse as sp
#
#     path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_EdgeList.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Edge list file not found: {path}")
#     logger.info("Loading weighted edge list from %s", path)
#
#     rows, cols, weights = [], [], []
#     with open(path, newline="") as f:
#         reader = csv.reader(f)
#         next(reader)  # header: Source,Target,Weight,...
#         for record in reader:
#             u, v, w = int(record[0]), int(record[1]), float(record[2])
#             rows.append(u)
#             cols.append(v)
#             weights.append(w)
#
#     # Symmetrize: the edge list stores each undirected edge once.
#     r = np.array(rows + cols)
#     c = np.array(cols + rows)
#     data = np.array(weights + weights, dtype=np.float64)
#     return sp.coo_matrix((data, (r, c)), shape=(n, n))
#
# The CSV matrix loader needs the node count, so load_original_network()
# becomes:
#     positions = _load_positions(set_name)
#     mat = _load_sparse_matrix(set_name, len(positions))


def _load_raw_image(set_name: str):
    path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"{set_name}_image.tif")
    if not os.path.exists(path):
        logger.warning("No image at %s. Returning None.", path)
        return None
    logger.info("Loading image from %s", path)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("Failed to load image: %s", path)
    return image
