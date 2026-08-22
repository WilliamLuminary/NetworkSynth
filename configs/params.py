from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class SynthParams:

    synthetic_frame_size: Tuple[int, int]
    closed_nodes_factor: float
    closed_edges_factor: float
    max_attempts: int
    # None = unseeded (non-reproducible).  Must be passed explicitly: there is
    # no default, so a caller cannot forget to state its intent.
    seed: Optional[int]

    @classmethod
    def from_config(cls, config) -> "SynthParams":
        """Build params from *config*, which must define every field.

        Raises ``AttributeError`` if the config is incomplete — that is
        deliberate.  Build this in the parent process and pass the result down;
        calling it inside a worker reintroduces the ``spawn`` problem above.
        """
        return cls(
            synthetic_frame_size=config.SYNTHETIC_FRAME_SIZE,
            closed_nodes_factor=config.CLOSED_NODES_FACTOR,
            closed_edges_factor=config.CLOSED_EDGES_FACTOR,
            max_attempts=config.MAX_ATTEMPTS,
            seed=config.SEED,
        )

    def for_worker(self, index: int) -> "SynthParams":
        """Return a copy carrying this worker's own derived seed.

        Every worker needs a *distinct* seed or all candidates come out
        identical, and a *deterministic* one or the run cannot be reproduced.
        Deriving ``SEED + index`` in the parent satisfies both, and keeps each
        worker's params self-describing.  Unseeded runs pass through unchanged.
        """
        if self.seed is None:
            return self
        return replace(self, seed=self.seed + index)
