# SPDX-License-Identifier: GPL-3.0-or-later
import os
from dataclasses import replace
from typing import Tuple

from ..base_config import BaseConfig
from ..dataset_id import DatasetId


class SnapshotConfig(BaseConfig):
    MODE = "generate"

    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (1530, 1530)

    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.3

    SNAPSHOT_INTERVAL = 0
    SELECT_BEST = 0

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 50
    ERROR_TOLERANCE = 0.15
    MEASURE_WEIGHTED = True

    RENDER_BFS_SNAPSHOT = replace(
        BaseConfig.RENDER_BFS_SNAPSHOT, node_size=0.5, line_width=0.5
    )
    RENDER_SYNTHETIC_GRAPH = replace(
        BaseConfig.RENDER_SYNTHETIC_GRAPH, node_size=0.5, line_width=0.5
    )

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "samples", "hybrid_mode")
