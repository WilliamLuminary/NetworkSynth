# src/configs/sweep_mode/config_sample.py
"""
Sweep mode default configuration (all A/B/C/D datasets).

Uses the same input data as hybrid mode:
    data/input/samples/hybrid_mode/
    ├── sample_{A,B,C,D}_pos.npy
    ├── sample_{A,B,C,D}_mat.npy
    └── sample_{A,B,C,D}_image.tif

Mechanically this *is* generate mode — same loaders, same BFS, same quality
gate — so it inherits the generate sample config and overrides only what
genuinely differs:

1. Node and edge factors are what the sweep **varies**, so they are deliberately
   not declared here: each trial supplies them as ``SynthParams`` (see
   ``pipelines/sweep.py``).  Declaring a value would be misleading, since every
   trial overrides it.
2. Saving is disabled and the per-trial network count is set at run time by
   ``pipelines/sweep.py``.
3. Input data lives under ``samples/hybrid_mode``, not ``samples/generate_mode``.

Point 3 is only safe because the inherited loaders resolve their directories
through ``cls``.  While they referenced the parent class by name, any subclass
silently read the parent's data.
"""

import logging
import os
from typing import Tuple

from ..base_config import BaseConfig
from ..enums import DatasetId
from ..generate_mode.config_sample import SampleConfig as GenerateSampleConfig

logger = logging.getLogger(__name__)


class SampleConfig(GenerateSampleConfig):
    """Sweep on all four hybrid-mode datasets."""

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
