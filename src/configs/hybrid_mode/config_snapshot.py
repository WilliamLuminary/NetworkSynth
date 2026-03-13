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

    SNAPSHOT_INTERVAL: int = 1
