# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from pathlib import Path

import pytest

import networksynth
from networksynth.configs.loader import load_config

_CONFIGS = Path(networksynth.__file__).resolve().parent / "configs"

pytestmark = pytest.mark.unit


def _config_modules_loaded():
    return {name for name in sys.modules if ".config_" in name}


class TestLoadingByPath:
    def test_it_returns_the_class_in_that_file(self):
        config = load_config(str(_CONFIGS / "generate_mode" / "config_snapshot_1x1.py"))

        assert config.__name__ == "Snapshot1x1Config"
        assert config.SNAPSHOT_INTERVAL == 10

    def test_the_path_has_to_be_the_real_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(str(_CONFIGS / "hybrid_mode" / "config_snapshot"))

    def test_an_absolute_path_works(self, tmp_path):
        absolute = str(_CONFIGS / "mosaic_mode" / "config_sample.py")

        assert load_config(absolute).__name__ == "SampleConfig"

    def test_a_missing_file_says_so(self):
        with pytest.raises(FileNotFoundError, match="No config file at"):
            load_config(str(_CONFIGS / "generate_mode" / "config_nope.py"))

    def test_a_config_outside_the_project_is_refused(self, tmp_path):
        stray = tmp_path / "config_stray.py"
        stray.write_text("class StrayConfig:\n    DATASETS = []\n")

        with pytest.raises(ValueError, match="outside the project"):
            load_config(str(stray))


class TestOnlyWhatIsAskedForLoads:
    def test_importing_configs_imports_no_config_module(self):
        before = _config_modules_loaded()

        import networksynth.configs  # noqa:F401

        assert _config_modules_loaded() == before

    def test_loading_one_does_not_drag_in_the_others(self):
        for stale in [n for n in sys.modules if "generate_mode.config_tmp" in n]:
            del sys.modules[stale]
        before = _config_modules_loaded()

        load_config(str(_CONFIGS / "generate_mode" / "config_tmp.py"))

        newly_loaded = _config_modules_loaded() - before
        assert newly_loaded <= {
            "networksynth.configs.generate_mode.config_tmp",
            "networksynth.configs.generate_mode.config_sample",
        }, newly_loaded


class TestOneConfigPerFile:
    def test_a_file_with_no_config_says_so(self, tmp_path, monkeypatch):
        module = type(sys)("networksynth.configs.fake_mode.config_empty")
        module.__name__ = "networksynth.configs.fake_mode.config_empty"

        from networksynth.configs.loader import config_class_in

        with pytest.raises(AttributeError, match="defines no config class"):
            config_class_in(module)

    def test_two_configs_in_one_file_is_ambiguous(self):
        module = type(sys)("networksynth.configs.fake_mode.config_two")

        class FirstConfig:
            DATASETS = []

        class SecondConfig:
            DATASETS = []

        FirstConfig.__module__ = module.__name__
        SecondConfig.__module__ = module.__name__
        module.FirstConfig = FirstConfig
        module.SecondConfig = SecondConfig

        from networksynth.configs.loader import config_class_in

        with pytest.raises(AttributeError, match="FirstConfig, SecondConfig"):
            config_class_in(module)


class TestEveryConfigInTheRepoLoads:

    @pytest.mark.parametrize(
        "path",
        sorted(
            str(p)
            for p in __import__("pathlib").Path("configs").glob("*_mode/config_*.py")
        ),
    )
    def test_it_loads(self, path):
        config = load_config(path)

        assert hasattr(config, "DATASETS")
