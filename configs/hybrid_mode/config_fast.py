# src/configs/hybrid_mode/config_fast.py
"""
Hybrid mode fast configuration.

Skips Phase 1 quality checking by setting ERROR_TOLERANCE to infinity
(first valid tile is accepted) and uses a small SYNTHETIC_FRAME_SIZE
so each seed tile is tiny.  Phase 2 then fills the full whiteboard
(whose size is based on IMAGE_SIZE × TARGET_SCALE).

Usage:
    python run.py hybrid --config fast
    python run.py hybrid --config fast --dataset A
"""
from typing import Tuple

from .config_sample import SampleConfig


class FastConfig(SampleConfig):
    """Hybrid mode with Phase 1 quality check bypassed."""

    # Accept the first valid tile immediately — no multifractal error filtering.
    ERROR_TOLERANCE = float("inf")
    MAX_ATTEMPTS = 1

    # Small frame so Phase 1 generates tiny seed tiles (just enough for
    # frontier nodes).  The whiteboard size is unaffected since it is
    # based on IMAGE_SIZE × TARGET_SCALE.
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (50, 50)
    MIN_TILE_NODES = 10

    # Fewer seed centers for a faster run.
    NUM_CENTERS: int = 1000
