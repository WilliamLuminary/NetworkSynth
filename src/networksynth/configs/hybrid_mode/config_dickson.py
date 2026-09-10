# SPDX-License-Identifier: GPL-3.0-or-later
import os
from dataclasses import replace
from typing import Dict, Tuple

from ..base_config import BaseConfig
from ..dataset_id import DatasetId


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
    IMAGE_SUFFIX = ".bmp"
