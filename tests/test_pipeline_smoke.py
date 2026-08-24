import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_fixture_data]


def _outputs(saver_dir: str) -> list:
    files = []
    for root, _, names in os.walk(saver_dir):
        files.extend(names)
    return files


def _tune(base, tmp_path, **overrides):
    config = type(f"Smoke{base.__name__}", (base,), {})
    config.BASE_OUTPUT_PATH = str(tmp_path)
    config.ERROR_CHECKER = "none"
    config.MAX_ATTEMPTS = 2
    config.LOG_MEMORY = False
    config.SEED = 1234
    for key, value in overrides.items():
        setattr(config, key, value)
    config.initialize()
    return config


# ------------------------------------------------------------------ #
# mosaic
# ------------------------------------------------------------------ #


class TestMosaicPipeline:
    def test_run_mosaic_for_dataset(self, tmp_path):
        from configs.mosaic_mode.config_sample import SampleConfig
        from handlers import create_run_paths
        from pipelines.mosaic import run_mosaic_for_dataset

        config = _tune(SampleConfig, tmp_path, GRID_ROWS=1, GRID_COLS=2)
        run_paths = create_run_paths(config)

        run_mosaic_for_dataset(config.get_datasets()[0], config, run_paths)

        assert _outputs(str(tmp_path)), "mosaic produced no output files"

    def test_compute_tile_layout_uses_its_config(self, tmp_path):
        from configs.mosaic_mode.config_sample import SampleConfig
        from pipelines.mosaic import compute_tile_layout

        config = _tune(SampleConfig, tmp_path, GRID_ROWS=1, GRID_COLS=2)

        tile_gen_frame, tile_offsets = compute_tile_layout(config)

        assert len(tile_offsets) == 2
        assert tile_gen_frame[0] > config.TILE_FRAME_SIZE[0]  # includes overlap


# ------------------------------------------------------------------ #
# scaling
# ------------------------------------------------------------------ #


class TestScalingPipeline:
    def test_run_scaling_for_dataset(self, tmp_path):
        from configs.scaling_mode.config_sample import SampleConfig
        from handlers import create_run_paths
        from pipelines.scaling import run_scaling_for_dataset

        config = _tune(
            SampleConfig,
            tmp_path,
            SCALE_ROWS=1,
            SCALE_COLS=2,
            MAX_GENERATION_ROUNDS=40,
        )
        run_paths = create_run_paths(config)

        run_scaling_for_dataset(config.get_datasets()[0], config, run_paths)

        assert _outputs(str(tmp_path)), "scaling produced no output files"


# ------------------------------------------------------------------ #
# hybrid
# ------------------------------------------------------------------ #


class TestHybridPipeline:
    @staticmethod
    def _config(tmp_path):
        from configs.hybrid_mode.config_sample import SampleConfig

        return _tune(
            SampleConfig,
            tmp_path,
            TARGET_SCALE=(1, 2),
            NUM_CENTERS=2,
            PHASE2_MAX_ROUNDS=10,
            TILE_FRAME_SIZE=(510, 510),
            SNAPSHOT_INTERVAL=0,
        )

    def test_run_hybrid_for_dataset(self, tmp_path):
        from handlers import create_run_paths
        from pipelines.hybrid import run_hybrid_for_dataset

        config = self._config(tmp_path)
        run_paths = create_run_paths(config)

        run_hybrid_for_dataset(config.get_datasets()[0], config, run_paths)

        assert _outputs(str(tmp_path)), "hybrid produced no output files"

    def test_compute_center_frames_uses_its_config(self, tmp_path):
        from pipelines.hybrid import compute_center_frames

        config = self._config(tmp_path)

        frames = compute_center_frames([(0.0, 0.0), (500.0, 500.0)], config)

        assert len(frames) == 2
        assert all(f == tuple(config.TILE_FRAME_SIZE) for f in frames)

    def test_apply_dataset_factors_returns_params_without_mutating(self, tmp_path):
        from configs import SynthParams
        from pipelines.hybrid import _apply_dataset_factors

        config = self._config(tmp_path)
        dataset_id = config.get_datasets()[0]
        config.DATASET_FACTORS = {dataset_id[0]: (1.9, 2.4)}
        before = config.CLOSED_NODES_FACTOR

        params = _apply_dataset_factors(
            dataset_id, config, SynthParams.from_config(config)
        )

        assert params.closed_nodes_factor == 1.9
        assert params.closed_edges_factor == 2.4
        assert config.CLOSED_NODES_FACTOR == before, "config must not be mutated"

    def test_apply_dataset_factors_passes_through_when_unset(self, tmp_path):
        from configs import SynthParams
        from pipelines.hybrid import _apply_dataset_factors

        config = self._config(tmp_path)
        config.DATASET_FACTORS = {}
        original = SynthParams.from_config(config)

        assert (
            _apply_dataset_factors(config.get_datasets()[0], config, original)
            is original
        )


# ------------------------------------------------------------------ #
# hybrid, with snapshots
# ------------------------------------------------------------------ #


class TestHybridSnapshots:
    """Snapshots are a config setting, not a pipeline of their own.

    ``hybrid_snapshot.py`` used to be a second copy of Phase 2 reached only by
    a runner script.  The same run is now ``hybrid`` with SNAPSHOT_INTERVAL
    set, so this asserts the images actually appear on that path.
    """

    def test_hybrid_writes_snapshots_when_the_config_asks_for_them(self, tmp_path):
        from configs.hybrid_mode.config_sample import SampleConfig
        from handlers import create_run_paths
        from pipelines.hybrid import run_hybrid_for_dataset

        config = _tune(
            SampleConfig,
            tmp_path,
            TARGET_SCALE=(1, 2),
            NUM_CENTERS=2,
            PHASE2_MAX_ROUNDS=6,
            TILE_FRAME_SIZE=(510, 510),
            # Larger than the round count, so exactly the first snapshot fires.
            SNAPSHOT_INTERVAL=100,
            HYBRID_SNAPSHOT_STYLE={"dpi": 72},
        )
        run_paths = create_run_paths(config)

        run_hybrid_for_dataset(config.get_datasets()[0], config, run_paths)

        outputs = _outputs(str(tmp_path))
        assert outputs, "hybrid produced no output files"
        assert any(
            name.startswith("snapshot_") for name in outputs
        ), f"SNAPSHOT_INTERVAL was set but nothing was rendered: {outputs}"


# ------------------------------------------------------------------ #
# sweep
# ------------------------------------------------------------------ #


class TestSweepPipeline:
    def test_generate_networks_threads_trial_params(self, tmp_path, monkeypatch):
        from analysis.error_checker import NullErrorChecker
        from configs.sweep_mode.config_sample import SampleConfig as SweepConfig
        from handlers import GenerationRun, create_run_paths
        from pipelines.sweep import generate_networks

        config = _tune(SweepConfig, tmp_path, SYNTHETIC_NETWORK_NUMBER=2)
        run_paths = create_run_paths(config)

        agent = GenerationRun(config, run_paths, config.get_datasets()[0])

        nodes_before = config.CLOSED_NODES_FACTOR
        avg_error, success_rate = generate_networks(
            agent, NullErrorChecker(), 1.7, 2.2, config
        )

        assert 0.0 <= success_rate <= 1.0
        assert avg_error is not None
        # The trial factors must NOT have been written into global config.
        assert config.CLOSED_NODES_FACTOR == nodes_before
