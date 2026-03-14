# src/configs/hybrid_mode/config_snapshot.py
"""
Hybrid mode snapshot configuration.

Identical to the default hybrid sample config but with Phase 2
round-by-round snapshot capture enabled.

Usage:
    python run.py hybrid --config snapshot
    python run.py hybrid --config snapshot --dataset A
"""
from .config_sample import SampleConfig


class SnapshotConfig(SampleConfig):
    """Hybrid mode with Phase 2 snapshots enabled."""

    SNAPSHOT_INTERVAL: int = 3

    # Visual style for hybrid Phase 2 snapshot PNGs.
    # Adjust these to control resolution and appearance.
    #   dpi        – None = auto (recommend_dpi based on node count)
    #   node_size  – scatter marker area in points²  (default 0.01)
    #   line_width – edge line width in points        (default 0.1)
    HYBRID_SNAPSHOT_STYLE: dict = {
        "dpi": 600,
        "node_size": 0.01,
        "line_width": 0.1,
    }
