from typing import Tuple

from .config_sample import SampleConfig


class FastConfig(SampleConfig):

    # Accept the first valid tile immediately — no error checking computed at all.
    ERROR_CHECKER = "none"
    MAX_ATTEMPTS = 10

    # Small frame so Phase 1 generates tiny seed tiles (just enough for
    # frontier nodes).  The whiteboard size is unaffected since it is
    # based on IMAGE_SIZE × TARGET_SCALE.
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (100, 100)
    MIN_TILE_NODES = 10

    # Fewer seed centers for a faster run.
    NUM_CENTERS: int = 1000
