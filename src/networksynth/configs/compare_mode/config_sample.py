# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_OUTPUT_PATH, CompareConfig
from ..dataset_id import DatasetId

_RESULTS = os.path.join(DEFAULT_OUTPUT_PATH, "latest_result", "sample_1")

CONFIG = CompareConfig(
    DATASETS=[DatasetId("analysis")],
    MEASURE_WEIGHTED=False,
    ORIGINAL_NETWORKS_PATH=os.path.join(_RESULTS, "original"),
    SYNTHETIC_NETWORKS_PATH=os.path.join(_RESULTS, "synthetic"),
)
