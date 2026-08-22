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
        pass

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

    def __init__(
        self,
        tolerance: float,
        measure_weighted: bool,
        full_q_band: bool,
    ) -> None:
        self.tolerance = tolerance
        self.measure_weighted = measure_weighted
        self.full_q_band = full_q_band
        self._ref_features = None

    def _analyzer(self, graph: SynthGraph):
        """Build an analyzer with this checker's captured settings.

        The checker is constructed in the parent and pickled to workers, so the
        settings travel with it.  Reading ``BaseConfig`` inside a worker would
        yield defaults under a ``spawn`` start method.
        """
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        return MultifractalAnalyzer(
            graph,
            measure_weighted=self.measure_weighted,
            full_q_band=self.full_q_band,
        )

    def compute_reference(self, graph: SynthGraph) -> None:
        self._ref_features = self._analyzer(graph).analyze_error_features()

    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        if self._ref_features is None:
            raise RuntimeError("compute_reference() must be called before check()")
        err_fea = self._analyzer(graph).analyze_error_features()
        error = MultifractalAnalyzer.analyze_error(err_fea, self._ref_features)
        return error < self.tolerance, error


class LengthAngleErrorChecker(ErrorChecker):
    """Quality gate on edge length and branching angle.

    Compares a candidate's mean edge length and mean angle between adjacent
    edges against the original network's, and passes when the mean relative
    error across those two is within *tolerance*.

    Cheaper and more local than the multifractal gate, which measures a global
    spectrum derived from all-pairs shortest paths.  This one says nothing about
    connectivity, so a candidate can pass here while being structurally quite
    unlike the original — it is a geometry gate, not a substitute.

    Both figures come from ``utils.compute_network_metrics``, which is also what
    ``generate_select`` ranks by, so the two agree by construction.  It computes
    three metrics this gate ignores; that waste is deliberate, because
    duplicating the angle geometry to avoid it would be the worse trade.
    """

    _KEYS = ("avg_length", "avg_angle")

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self._reference = None

    @classmethod
    def _measure(cls, graph: SynthGraph) -> dict:
        from utils import compute_network_metrics

        metrics = compute_network_metrics(graph)
        return {key: metrics[key] for key in cls._KEYS}

    def compute_reference(self, graph: SynthGraph) -> None:
        self._reference = self._measure(graph)

    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        from utils import metric_distance

        if self._reference is None:
            raise RuntimeError("compute_reference() must be called before check()")
        error = metric_distance(self._measure(graph), self._reference)
        return error < self.tolerance, error


_CHECKERS = {
    "none": lambda cfg: NullErrorChecker(),
    "multifractal": lambda cfg: MultifractalErrorChecker(
        cfg.ERROR_TOLERANCE,
        measure_weighted=cfg.MEASURE_WEIGHTED,
        full_q_band=cfg.FULL_Q_BAND,
    ),
    "length_angle": lambda cfg: LengthAngleErrorChecker(cfg.ERROR_TOLERANCE),
}


def create_error_checker(config) -> ErrorChecker:
    """Build the checker selected by ``config.ERROR_CHECKER``.

    The config is passed in rather than imported so this module holds no
    dependency on the global ``BaseConfig`` namespace.

    To add an algorithm: implement an :class:`ErrorChecker` subclass and
    register a builder in ``_CHECKERS`` keyed by its config name.  Each
    builder receives the config so it can read whatever parameters its
    algorithm needs.
    """
    name = config.ERROR_CHECKER
    builder = _CHECKERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown ERROR_CHECKER {name!r}; available: {sorted(_CHECKERS)}"
        )
    return builder(config)
