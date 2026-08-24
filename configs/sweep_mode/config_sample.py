import logging
import os
from typing import Tuple

from ..base_config import BaseConfig
from ..dataset_id import DatasetId
from ..generate_mode.config_sample import SampleConfig as GenerateSampleConfig

logger = logging.getLogger(__name__)


class SampleConfig(GenerateSampleConfig):
    MODE = "sweep"

    DATASETS = [
        DatasetId("sample_A"),
        DatasetId("sample_B"),
        DatasetId("sample_C"),
        DatasetId("sample_D"),
    ]

    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    MEASURE_WEIGHTED = True

    # A sweep scores factor combinations; the networks themselves are throwaway.
    DISABLE_SAVING = True
    DISABLE_SAVING_NOTE = "Sweeping Experiment"

    # Sweep ranges, previously hardcoded as DEFAULT_NF_RANGE / DEFAULT_EF_RANGE
    # in pipelines/sweep.py.
    NF_RANGE: Tuple[float, float] = (0.3, 2.0)
    EF_RANGE: Tuple[float, float] = (0.3, 2.0)

    BASE_INPUT_PATH = os.path.join(
        BaseConfig.BASE_INPUT_PATH,
        "samples",
        "hybrid_mode",
    )
    # Re-declared because the parent derives these from *its* BASE_INPUT_PATH at
    # class-body time; overriding BASE_INPUT_PATH alone would not move them.
    POSITION_DATA_DIR = BASE_INPUT_PATH
    ADJ_MATRIX_DATA_DIR = BASE_INPUT_PATH
    IMAGES_DIR = BASE_INPUT_PATH
