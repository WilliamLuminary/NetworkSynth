# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple

from .config_sample import SampleConfig


class FastConfig(SampleConfig):

    ERROR_CHECKER = "none"
    MAX_ATTEMPTS = 10

    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (100, 100)
    MIN_TILE_NODES = 10

    NUM_CENTERS: int = 1000
