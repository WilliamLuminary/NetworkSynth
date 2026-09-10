# SPDX-License-Identifier: GPL-3.0-or-later
import os
from typing import Dict, Tuple

from ..base_config import BaseConfig
from ..dataset_id import DatasetId
from ..file_definitions import SYNTHETIC_DIR, SaveSpec, save_png, save_webp


class SampleConfig(BaseConfig):
    MODE = "hybrid"

    LOG_MEMORY = True

    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    TARGET_SCALE: Tuple[int, int] = (100, 100)

    NUM_CENTERS: int = 2000
    PHASE2_MAX_ROUNDS: int = 500
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5

    TILE_FRAME_SIZE: Tuple[int, int] = None
    TILE_FRAME_FACTOR: float = 0.5

    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    CLOSED_NODES_FACTOR = 1.4
    CLOSED_EDGES_FACTOR = 2.0

    DATASET_FACTORS: Dict[str, Tuple[float, float]] = {
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

    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "hybrid_mode",
    )

    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "png", save_png),
    )
