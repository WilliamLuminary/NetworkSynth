# SPDX-License-Identifier: GPL-3.0-or-later
import os
from dataclasses import replace

import pytest

from networksynth.configs import RenderStyle

pytestmark = [pytest.mark.integration, pytest.mark.requires_fixture_data]


def _outputs(saver_dir: str) -> list:
    files = []
    for root, _, names in os.walk(saver_dir):
        files.extend(names)
    return files


def _tune(base, tmp_path, **overrides):
    return replace(
        base,
        BASE_OUTPUT_PATH=str(tmp_path),
        ERROR_CHECKER="none",
        MAX_ATTEMPTS=2,
        LOG_MEMORY=False,
        SEED=1234,
        **overrides,
    )


class TestHybridPipeline:
    @staticmethod
    def _config(tmp_path):
        from tests.fixture_config import FIXTURE_HYBRID

        return _tune(FIXTURE_HYBRID, tmp_path)

    def test_run_hybrid_for_dataset(self, tmp_path):
        from networksynth.handlers import create_run_paths
        from networksynth.pipelines.hybrid import run_hybrid_for_dataset

        config = self._config(tmp_path)
        run_paths = create_run_paths(config)

        run_hybrid_for_dataset(config.DATASETS[0], config, run_paths)

        assert _outputs(str(tmp_path)), "hybrid produced no output files"

    def test_compute_center_frames_uses_its_config(self, tmp_path):
        from networksynth.pipelines.hybrid import compute_center_frames

        config = self._config(tmp_path)

        frames = compute_center_frames([(0.0, 0.0), (500.0, 500.0)], config)

        assert len(frames) == 2
        assert all(f == tuple(config.TILE_FRAME_SIZE) for f in frames)

    def test_apply_dataset_factors_returns_params_without_mutating(self, tmp_path):
        from networksynth.configs import SynthParams
        from networksynth.pipelines.hybrid import _apply_dataset_factors

        dataset_id = self._config(tmp_path).DATASETS[0]
        config = replace(
            self._config(tmp_path), DATASET_FACTORS={dataset_id[0]: (1.9, 2.4)}
        )
        before = config.CLOSED_NODES_FACTOR

        params = _apply_dataset_factors(
            dataset_id, config, SynthParams.from_config(config)
        )

        assert params.closed_nodes_factor == 1.9
        assert params.closed_edges_factor == 2.4
        assert config.CLOSED_NODES_FACTOR == before, "config must not be mutated"

    def test_apply_dataset_factors_passes_through_when_unset(self, tmp_path):
        from networksynth.configs import SynthParams
        from networksynth.pipelines.hybrid import _apply_dataset_factors

        config = replace(self._config(tmp_path), DATASET_FACTORS={})
        original = SynthParams.from_config(config)

        assert _apply_dataset_factors(config.DATASETS[0], config, original) is original


class TestHybridSnapshots:

    def test_hybrid_writes_snapshots_when_the_config_asks_for_them(self, tmp_path):
        from networksynth.handlers import create_run_paths
        from networksynth.pipelines.hybrid import run_hybrid_for_dataset
        from tests.fixture_config import FIXTURE_HYBRID

        config = _tune(
            FIXTURE_HYBRID,
            tmp_path,
            PHASE2_MAX_ROUNDS=6,
            SNAPSHOT_INTERVAL=100,
            RENDER_HYBRID_SNAPSHOT=RenderStyle(dpi=72),
        )
        run_paths = create_run_paths(config)

        run_hybrid_for_dataset(config.DATASETS[0], config, run_paths)

        outputs = _outputs(str(tmp_path))
        assert outputs, "hybrid produced no output files"
        assert any(
            name.startswith("snapshot_") for name in outputs
        ), f"SNAPSHOT_INTERVAL was set but nothing was rendered: {outputs}"


class TestSweepPipeline:
    def test_generate_networks_threads_trial_params(self, tmp_path, monkeypatch):
        from networksynth.analysis.error_checker import NullErrorChecker
        from networksynth.handlers import GenerationRun, create_run_paths
        from networksynth.pipelines.sweep import generate_networks
        from tests.fixture_config import FIXTURE_SWEEP

        config = _tune(FIXTURE_SWEEP, tmp_path, SYNTHETIC_NETWORK_NUMBER=2)
        run_paths = create_run_paths(config)

        agent = GenerationRun(config, run_paths, config.DATASETS[0])

        avg_error, success_rate = generate_networks(
            agent, NullErrorChecker(), 1.7, 2.2, config
        )

        assert 0.0 <= success_rate <= 1.0
        assert avg_error is not None
