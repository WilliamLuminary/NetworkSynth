# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import replace

from .config_sample import CONFIG as SAMPLE

CONFIG = replace(
    SAMPLE,
    SNAPSHOT_INTERVAL=-60,
    RENDER_HYBRID_SNAPSHOT=replace(
        SAMPLE.RENDER_HYBRID_SNAPSHOT, dpi=600, node_size=0.01, line_width=0.1
    ),
)
