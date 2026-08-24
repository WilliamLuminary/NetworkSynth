from dataclasses import replace

from .config_sample import SampleConfig


class SnapshotConfig(SampleConfig):
    # Snapshot schedule:
    #   N > 0  →  linear: every N rounds (e.g. 3 = rounds 3, 6, 9, ...)
    #   N < 0  →  log-spaced: ~|N| total snapshots (e.g. -60 = ~60 snapshots)
    SNAPSHOT_INTERVAL: int = -60

    # At 600 dpi both sizes land below one pixel, so these draw the thinnest
    # line and a single-pixel node.
    RENDER_HYBRID_SNAPSHOT = replace(
        SampleConfig.RENDER_HYBRID_SNAPSHOT, dpi=600, node_size=0.01, line_width=0.1
    )
