# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
from typing import List  # noqa:F401

from ..base_config import BaseConfig
from ..dataset_id import DatasetId


class SampleConfig(BaseConfig):
    MODE = "compare"

    MEASURE_WEIGHTED = False

    DATASETS = [DatasetId("analysis")]

    _RESULTS = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "latest_result", "sample_1")
    ORIGINAL_NETWORKS_PATH = os.path.join(_RESULTS, "original")
    SYNTHETIC_NETWORKS_PATH = os.path.join(_RESULTS, "synthetic")
