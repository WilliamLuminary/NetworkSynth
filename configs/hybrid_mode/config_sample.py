import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from graphs.synth_graph import SynthGraph
from utils import resize_image, transpose_positions, trim_image

from ..base_config import BaseConfig
from ..dataset_id import DatasetId
from ..file_definitions import SYNTHETIC_DIR, SaveSpec, save_png, save_webp

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    MODE = "hybrid"

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
    # nearest-neighbor distance * TILE_FRAME_FACTOR. Factor 0.5 leaves a gap
    # (~half the spacing) for Phase 2 to stitch; smaller factor = wider gap.
    # The floor is the same rule at the closest spacing the sampler allows
    # (MIN_CENTER_DISTANCE_FACTOR * max(FRAME_SIZE) * TILE_FRAME_FACTOR), so
    # there is no pixel count here to keep in step with the frame.
    TILE_FRAME_SIZE: Tuple[int, int] = None
    TILE_FRAME_FACTOR: float = 0.5

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

    # --- Input paths ---
    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "hybrid_mode",
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
        image = resize_image(image, SampleConfig.FRAME_SIZE)
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
