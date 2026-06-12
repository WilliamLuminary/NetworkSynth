# src/analysis/error_checker.py
"""
Pluggable error-checking abstraction for network generation pipelines.

Provides a common interface so that different quality-gate strategies
(multifractal, topological, etc.) can be swapped without touching the
generation loop.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


class ErrorChecker(ABC):
    """Abstract quality gate evaluated during network generation.

    Subclasses must implement two methods:

    * ``compute_reference`` — called once on the original network to
      establish the baseline that generated candidates are compared to.
    * ``check`` — called on every candidate to decide whether it passes
      the quality gate.
    """

    @abstractmethod
    def compute_reference(self, graph: SynthGraph) -> None:
        """Compute and store reference features from the original network."""

    @abstractmethod
    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        """Evaluate a candidate network.

        Returns
        -------
        passed : bool
            ``True`` if the candidate meets the quality threshold.
        error : float
            Numeric error score (lower is better).  Implementations that
            do not compute an error should return ``0.0``.
        """


class NullErrorChecker(ErrorChecker):
    """No-op checker — every candidate passes with zero error."""

    def compute_reference(self, graph: SynthGraph) -> None:
        pass

    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        return True, 0.0


class MultifractalErrorChecker(ErrorChecker):
    """Quality gate based on multifractal spectrum distance.

    Compares the Holder exponent and spectrum width of each candidate
    against those of the original network.  A candidate passes when
    the Euclidean distance between feature vectors is below *tolerance*.
    """

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self._ref_features = None

    def compute_reference(self, graph: SynthGraph) -> None:
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        self._ref_features = MultifractalAnalyzer(graph).analyze_error_features()

    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        if self._ref_features is None:
            raise RuntimeError("compute_reference() must be called before check()")
        err_fea = MultifractalAnalyzer(graph).analyze_error_features()
        error = MultifractalAnalyzer.analyze_error(err_fea, self._ref_features)
        return error < self.tolerance, error


_CHECKERS = {
    "none": lambda cfg: NullErrorChecker(),
    "multifractal": lambda cfg: MultifractalErrorChecker(cfg.ERROR_TOLERANCE),
}


def create_error_checker() -> ErrorChecker:
    """Build the checker selected by ``BaseConfig.ERROR_CHECKER``.

    To add an algorithm: implement an :class:`ErrorChecker` subclass and
    register a builder in ``_CHECKERS`` keyed by its config name.  Each
    builder receives the config so it can read whatever parameters its
    algorithm needs.
    """
    from configs import BaseConfig

    name = BaseConfig.ERROR_CHECKER
    builder = _CHECKERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown ERROR_CHECKER {name!r}; available: {sorted(_CHECKERS)}"
        )
    return builder(BaseConfig)
