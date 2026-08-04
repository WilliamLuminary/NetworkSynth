# src/configs/params.py
"""Immutable generation parameters.

``BaseConfig`` publishes its values globally via ``_inject_dependencies()``,
which only reaches child processes under a ``fork`` start method.  On ``spawn``
(the default on macOS and Windows) children re-import ``configs`` unmutated and
read wrong — or missing — values.

``SynthParams`` carries the values that are read *inside* worker processes.  It
is frozen and picklable, so the parent builds it once and passes it explicitly
to each worker instead of relying on inherited class state.

**No field has a default and no consumer accepts ``None``.**  A misconfigured
run must fail immediately rather than proceed on a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SynthParams:
    """Generation parameters needed by worker processes."""

    synthetic_frame_size: Tuple[int, int]
    closed_nodes_factor: float
    closed_edges_factor: float
    max_attempts: int

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
        )
