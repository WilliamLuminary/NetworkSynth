from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple

from graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


class ErrorChecker(ABC):

    @abstractmethod
    def compute_reference(self, graph: SynthGraph) -> None:
        pass

    @abstractmethod
    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        pass


class NullErrorChecker(ErrorChecker):

    def compute_reference(self, graph: SynthGraph) -> None:
        pass

    def check(self, graph: SynthGraph) -> Tuple[bool, float]:
        return True, 0.0


class MultifractalErrorChecker(ErrorChecker):

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
    name = config.ERROR_CHECKER
    builder = _CHECKERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown ERROR_CHECKER {name!r}; available: {sorted(_CHECKERS)}"
        )
    return builder(config)
