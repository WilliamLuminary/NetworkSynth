# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

from .config_sample import SampleConfig


class SnapshotConfig(SampleConfig):
    SNAPSHOT_INTERVAL: int = -60

    RENDER_HYBRID_SNAPSHOT = replace(
        SampleConfig.RENDER_HYBRID_SNAPSHOT, dpi=600, node_size=0.01, line_width=0.1
    )
