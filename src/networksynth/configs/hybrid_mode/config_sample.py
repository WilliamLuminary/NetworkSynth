# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_INPUT_PATH, HybridConfig
from ..dataset_id import DatasetId

CONFIG = HybridConfig(
    DATASETS=[
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ],
    BASE_INPUT_PATH=os.path.join(DEFAULT_INPUT_PATH, "samples", "hybrid_mode"),
    LOG_MEMORY=True,
    TARGET_SCALE=(100, 100),
    NUM_CENTERS=2000,
    MIN_CENTER_DISTANCE_FACTOR=1.5,
    TILE_FRAME_FACTOR=0.5,
    FRAME_SIZE=(510, 510),
    SYNTHETIC_FRAME_SIZE=(510, 510),
    CLOSED_NODES_FACTOR=1.4,
    CLOSED_EDGES_FACTOR=2.0,
    DATASET_FACTORS={
        "sample_A": (1.0, 1.4),
        "sample_B": (1.2, 0.9),
        "sample_C": (1.0, 1.2),
        "sample_D": (1.2, 0.9),
    },
    MAX_ATTEMPTS=50,
    MEASURE_WEIGHTED=True,
)
