# tests/analysis/test_error_checker.py
"""
Unit tests for the error-checker settings handoff (analysis/error_checker.py).

``MultifractalErrorChecker`` is built in the parent process and pickled to
workers, so it must carry ``measure_weighted`` / ``full_q_band`` with it.  A
worker that instead read ``BaseConfig`` would get defaults under a ``spawn``
start method, silently changing the quality gate.
"""

import pickle

import pytest

from analysis.error_checker import (
    MultifractalErrorChecker,
    NullErrorChecker,
    create_error_checker,
)
from configs import BaseConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def base_config_values():
    saved = {
        name: getattr(BaseConfig, name, "__missing__")
        for name in (
            "ERROR_CHECKER",
            "ERROR_TOLERANCE",
            "MEASURE_WEIGHTED",
            "FULL_Q_BAND",
        )
    }
    yield
    for name, value in saved.items():
        if value == "__missing__":
            if hasattr(BaseConfig, name):
                delattr(BaseConfig, name)
        else:
            setattr(BaseConfig, name, value)


class TestSettingsCapture:
    def test_settings_are_required(self):
        """No defaults: an under-specified checker must fail immediately."""
        with pytest.raises(TypeError):
            MultifractalErrorChecker(0.15)

    def test_captures_explicit_settings(self):
        checker = MultifractalErrorChecker(0.2, measure_weighted=True, full_q_band=True)

        assert checker.tolerance == 0.2
        assert checker.measure_weighted is True
        assert checker.full_q_band is True

    def test_settings_survive_a_pickle_round_trip(self):
        """This is how the checker reaches a worker under spawn."""
        checker = MultifractalErrorChecker(0.3, measure_weighted=True, full_q_band=True)

        revived = pickle.loads(pickle.dumps(checker))

        assert revived.tolerance == 0.3
        assert revived.measure_weighted is True
        assert revived.full_q_band is True


class TestCreateErrorChecker:
    def test_reads_settings_from_base_config(self, base_config_values):
        BaseConfig.ERROR_CHECKER = "multifractal"
        BaseConfig.ERROR_TOLERANCE = 0.11
        BaseConfig.MEASURE_WEIGHTED = True
        BaseConfig.FULL_Q_BAND = True

        checker = create_error_checker(BaseConfig)

        assert isinstance(checker, MultifractalErrorChecker)
        assert checker.tolerance == 0.11
        assert checker.measure_weighted is True
        assert checker.full_q_band is True

    def test_none_selects_the_null_checker(self, base_config_values):
        BaseConfig.ERROR_CHECKER = "none"

        assert isinstance(create_error_checker(BaseConfig), NullErrorChecker)

    def test_unknown_name_raises(self, base_config_values):
        BaseConfig.ERROR_CHECKER = "nope"

        with pytest.raises(ValueError, match="Unknown ERROR_CHECKER"):
            create_error_checker(BaseConfig)


class TestAnalyzerHonoursExplicitSettings:
    """Explicit arguments must beat BaseConfig — that is the spawn guarantee."""

    def test_full_q_band_argument_wins(
        self, base_config_values, load_unweighted_test_synth_graph
    ):
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        BaseConfig.MEASURE_WEIGHTED = False
        BaseConfig.FULL_Q_BAND = False

        assert (
            MultifractalAnalyzer(
                load_unweighted_test_synth_graph,
                measure_weighted=False,
                full_q_band=True,
            )._full_q_band
            is True
        )

    def test_settings_are_required(self, load_unweighted_test_synth_graph):
        """No BaseConfig fallback: omitting them must fail immediately."""
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        with pytest.raises(TypeError):
            MultifractalAnalyzer(load_unweighted_test_synth_graph)

    def test_weighted_setting_ignored_for_unweighted_graph(
        self, load_unweighted_test_synth_graph
    ):
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        analyzer = MultifractalAnalyzer(
            load_unweighted_test_synth_graph, measure_weighted=True, full_q_band=False
        )

        assert analyzer.weighted is False
