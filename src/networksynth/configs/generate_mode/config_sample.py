# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_INPUT_PATH, GenerateConfig
from ..dataset_id import DatasetId

CONFIG = GenerateConfig(
    DATASETS=[DatasetId("sample_1"), DatasetId("sample_2"), DatasetId("sample_3")],
    BASE_INPUT_PATH=os.path.join(DEFAULT_INPUT_PATH, "samples", "generate_mode"),
    FRAME_SIZE=(1887 // 4, 2048 // 4),
    SYNTHETIC_FRAME_SIZE=(1887 // 4, 2048 // 4),
    CLOSED_NODES_FACTOR=1.2,
    CLOSED_EDGES_FACTOR=0.8,
    MEASURE_WEIGHTED=False,
)
