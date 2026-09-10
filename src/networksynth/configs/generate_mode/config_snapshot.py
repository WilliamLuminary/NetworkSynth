# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_INPUT_PATH, GenerateConfig
from ..dataset_id import DatasetId
from ..file_definitions import RenderStyle

CONFIG = GenerateConfig(
    DATASETS=[
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ],
    BASE_INPUT_PATH=os.path.join(DEFAULT_INPUT_PATH, "samples", "hybrid_mode"),
    FRAME_SIZE=(510, 510),
    SYNTHETIC_FRAME_SIZE=(1530, 1530),
    CLOSED_NODES_FACTOR=1.0,
    CLOSED_EDGES_FACTOR=1.3,
    SYNTHETIC_GRAPH_NUMBER=1,
    MAX_ATTEMPTS=50,
    MEASURE_WEIGHTED=True,
    RENDER_BFS_SNAPSHOT=RenderStyle(node_size=0.5, line_width=0.5, dpi=300),
    RENDER_SYNTHETIC_GRAPH=RenderStyle(
        node_size=0.5, line_width=0.5, dpi=300, show_on_the_fly=False
    ),
)
