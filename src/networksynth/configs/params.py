# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class SynthParams:
    synthetic_frame_size: Tuple[int, int]
    closed_nodes_factor: float
    closed_edges_factor: float
    max_attempts: int
    seed: Optional[int]

    @classmethod
    def from_config(cls, config) -> "SynthParams":
        return cls(
            synthetic_frame_size=config.SYNTHETIC_FRAME_SIZE,
            closed_nodes_factor=config.CLOSED_NODES_FACTOR,
            closed_edges_factor=config.CLOSED_EDGES_FACTOR,
            max_attempts=config.MAX_ATTEMPTS,
            seed=config.SEED,
        )

    def for_worker(self, index: int) -> "SynthParams":
        if self.seed is None:
            return self
        return replace(self, seed=self.seed + index)
