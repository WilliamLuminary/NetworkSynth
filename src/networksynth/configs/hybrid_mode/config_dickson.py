# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_INPUT_PATH, HybridConfig
from ..dataset_id import DatasetId
from ..file_definitions import RenderStyle

CONFIG = HybridConfig(
    DATASETS=[
        DatasetId("gel1"),
        DatasetId("gel2"),
        DatasetId("gel3"),
        DatasetId("gel4"),
    ],
    BASE_INPUT_PATH=os.path.join(DEFAULT_INPUT_PATH, "dickson"),
    IMAGE_SUFFIX=".bmp",
    LOG_MEMORY=True,
    TARGET_SCALE=(40, 40),
    NUM_CENTERS=500,
    MIN_CENTER_DISTANCE_FACTOR=1.5,
    TILE_FRAME_FACTOR=0.5,
    FRAME_SIZE=(1030, 730),
    SYNTHETIC_FRAME_SIZE=(1030, 730),
    CLOSED_NODES_FACTOR=1.2,
    CLOSED_EDGES_FACTOR=1.5,
    MAX_ATTEMPTS=5,
    MEASURE_WEIGHTED=True,
    RENDER_ORIGINAL_GRAPH=RenderStyle(node_size=3.0, line_width=3.0, dpi=300),
    RENDER_HYBRID_GRAPH=RenderStyle(max_px=8000),
    RENDER_HYBRID_SNAPSHOT=RenderStyle(max_px=8000),
)
