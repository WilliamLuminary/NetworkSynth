# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

from .config_sample import CONFIG as SAMPLE

CONFIG = replace(
    SAMPLE,
    ERROR_CHECKER="none",
    MAX_ATTEMPTS=10,
    SYNTHETIC_FRAME_SIZE=(100, 100),
    MIN_TILE_NODES=10,
    NUM_CENTERS=1000,
)
