# src/configs/hybrid_mode/config_fast.py
"""
Hybrid mode fast configuration.

Skips Phase 1 quality checking by setting ERROR_TOLERANCE to infinity
(first valid tile is accepted) and reduces the number of seed centers
for quicker runs.

Usage:
    python run.py hybrid --config fast
    python run.py hybrid --config fast --dataset A
"""
from .config_sample import SampleConfig


class FastConfig(SampleConfig):
    """Hybrid mode with Phase 1 quality check bypassed."""

    # Accept the first valid tile immediately — no multifractal error filtering.
    ERROR_TOLERANCE = float("inf")
    MAX_ATTEMPTS = 1

    # Fewer seed centers for a faster run.
    NUM_CENTERS: int = 1000
