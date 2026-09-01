# SPDX-License-Identifier: GPL-3.0-or-later
import pickle

import numpy as np
import pytest

from analysis.error_checker import (
    MultifractalErrorChecker,
    NullErrorChecker,
    create_error_checker,
)
from configs import BaseConfig
from graphs.synth_graph import SynthGraph

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
        with pytest.raises(TypeError):
            MultifractalErrorChecker(0.15)

    def test_captures_explicit_settings(self):
        checker = MultifractalErrorChecker(0.2, measure_weighted=True, full_q_band=True)

        assert checker.tolerance == 0.2
        assert checker.measure_weighted is True
        assert checker.full_q_band is True

    def test_settings_survive_a_pickle_round_trip(self):
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
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        with pytest.raises(TypeError):
            MultifractalAnalyzer(load_unweighted_test_synth_graph)

    def test_weighted_setting_rejected_for_unweighted_graph(
        self, load_unweighted_test_synth_graph
    ):
        from analysis.multifractal_analyzer import MultifractalAnalyzer

        with pytest.raises(AssertionError, match="carries no edge weights"):
            MultifractalAnalyzer(
                load_unweighted_test_synth_graph,
                measure_weighted=True,
                full_q_band=False,
            )


class TestLengthAngleErrorChecker:

    @staticmethod
    def _lattice(side=8, spacing=10.0, seed=1):
        import igraph as ig

        rng = np.random.default_rng(seed)
        coords = [(x * spacing, y * spacing) for y in range(side) for x in range(side)]
        positions = np.asarray(coords, dtype=float)
        positions += rng.normal(0, spacing * 0.02, size=positions.shape)
        pairs = []
        for y in range(side):
            for x in range(side):
                here = y * side + x
                if x + 1 < side:
                    pairs.append((here, here + 1))
                if y + 1 < side:
                    pairs.append((here, here + side))
        return SynthGraph(ig.Graph(n=len(coords), edges=pairs), positions)

    def _checker(self, tolerance=0.15):
        from analysis.error_checker import LengthAngleErrorChecker

        return LengthAngleErrorChecker(tolerance)

    def test_it_is_reachable_through_the_registry(self):
        from analysis.error_checker import LengthAngleErrorChecker, create_error_checker
        from configs import BaseConfig

        class Config(BaseConfig):
            ERROR_CHECKER = "length_angle"
            ERROR_TOLERANCE = 0.2

        checker = create_error_checker(Config)

        assert isinstance(checker, LengthAngleErrorChecker)
        assert checker.tolerance == 0.2

    def test_check_before_reference_raises(self):
        with pytest.raises(RuntimeError, match="compute_reference"):
            self._checker().check(self._lattice())

    def test_an_identical_graph_scores_zero(self):
        graph = self._lattice()
        checker = self._checker()
        checker.compute_reference(graph)

        passed, error = checker.check(graph)

        assert passed
        assert error == pytest.approx(0.0, abs=1e-12)

    def test_it_measures_only_length_and_angle(self):
        checker = self._checker(tolerance=0.05)
        checker.compute_reference(self._lattice(side=6))

        passed, error = checker.check(self._lattice(side=12))

        assert set(checker._reference) == {"avg_length", "avg_angle"}
        assert passed, f"size should not matter, error was {error}"

    def test_different_geometry_fails(self):
        checker = self._checker(tolerance=0.15)
        checker.compute_reference(self._lattice(spacing=10.0))

        passed, error = checker.check(self._lattice(spacing=100.0))

        assert not passed
        assert error > 0.15

    def test_tolerance_decides_the_boundary(self):
        checker_tight = self._checker(tolerance=0.001)
        checker_loose = self._checker(tolerance=10.0)
        reference = self._lattice(spacing=10.0)
        candidate = self._lattice(spacing=12.0)
        for checker in (checker_tight, checker_loose):
            checker.compute_reference(reference)

        assert not checker_tight.check(candidate)[0]
        assert checker_loose.check(candidate)[0]

    def test_it_survives_pickling(self):
        import pickle

        checker = self._checker()
        checker.compute_reference(self._lattice())

        restored = pickle.loads(pickle.dumps(checker))

        assert restored._reference == checker._reference
        assert restored.check(self._lattice())[0]
