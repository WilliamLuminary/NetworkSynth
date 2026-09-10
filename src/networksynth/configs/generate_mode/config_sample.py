# SPDX-License-Identifier: GPL-3.0-or-later
import os
from typing import List

from ..base_config import BaseConfig
from ..dataset_id import DatasetId


def _generate_sample_datasets() -> List[DatasetId]:
    return [DatasetId("sample_1"), DatasetId("sample_2"), DatasetId("sample_3")]


class SampleConfig(BaseConfig):
    MODE = "generate"

    DATASETS = _generate_sample_datasets()

    IMAGE_SIZE = (1887, 2048)
    FRAME_SIZE = (1887 // 4, 2048 // 4)
    SYNTHETIC_FRAME_SIZE = FRAME_SIZE
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH, "samples", "generate_mode"
    )
