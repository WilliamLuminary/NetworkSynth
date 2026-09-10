# SPDX-License-Identifier: GPL-3.0-or-later
import os

from ..base_config import DEFAULT_INPUT_PATH, SweepConfig
from ..dataset_id import DatasetId

CONFIG = SweepConfig(
    DATASETS=[
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ],
    BASE_INPUT_PATH=os.path.join(DEFAULT_INPUT_PATH, "samples", "hybrid_mode"),
    FRAME_SIZE=(510, 510),
    SYNTHETIC_FRAME_SIZE=(510, 510),
    MEASURE_WEIGHTED=True,
    DISABLE_SAVING=True,
    DISABLE_SAVING_NOTE="Sweeping Experiment",
    NF_RANGE=(0.3, 2.0),
    EF_RANGE=(0.3, 2.0),
)
