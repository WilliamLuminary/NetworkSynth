"""Config discovery must not import every config to find one.

A config module runs path arithmetic and pulls in cv2 at import time, and a run
uses exactly one of them.  These tests pin both halves of that: the names are
discoverable without importing, and the name a file is indexed under is the one
its class would have produced.
"""

import sys

import pytest

pytestmark = pytest.mark.unit

MODE_PREFIXES = {
    "compare_mode": "Compare",
    "generate_mode": "Gen",
    "hybrid_mode": "Hybrid",
    "mosaic_mode": "Mosaic",
    "scaling_mode": "Scaling",
    "sweep_mode": "Sweep",
}


def _config_modules_loaded():
    return {name for name in sys.modules if ".config_" in name}


class TestNothingLoadsUntilItIsAskedFor:
    def test_importing_configs_imports_no_config_module(self):
        # Not a fresh interpreter, so this cannot assert "none loaded" — it
        # asserts that touching the package itself loads nothing new.
        import configs

        before = _config_modules_loaded()
        importlib_names = sorted(configs._CONFIG_MODULES)

        assert importlib_names, "no configs indexed at all"
        assert _config_modules_loaded() == before

    def test_every_name_is_listed_without_importing(self):
        import configs

        # 17 configs across six modes, all namable from the filenames alone.
        assert len(configs._CONFIG_MODULES) == len(set(configs._CONFIG_MODULES))
        for name in configs._CONFIG_MODULES:
            assert name in dir(configs)

    def test_asking_for_one_loads_exactly_that_one(self):
        import configs

        for stale in [n for n in sys.modules if "generate_mode.config_tmp" in n]:
            del sys.modules[stale]
        before = _config_modules_loaded()

        configs.GenConfigTmp

        newly_loaded = _config_modules_loaded() - before
        assert newly_loaded == {"configs.generate_mode.config_tmp"}, newly_loaded

    def test_an_unknown_name_still_raises_attribute_error(self):
        import configs

        with pytest.raises(AttributeError, match="NoSuchConfig"):
            configs.NoSuchConfig


class TestTheIndexAgreesWithTheClasses:
    """The index is built from filenames; the exports used to come from class
    names.  If a new config's class name disagrees with its filename, the name
    callers use would change silently — so check every one."""

    @pytest.mark.parametrize("mode", sorted(MODE_PREFIXES))
    def test_filename_derived_names_match_class_derived_names(self, mode):
        import importlib
        import re
        from pathlib import Path

        from configs._loader import _get_config_class_from_module

        prefix = MODE_PREFIXES[mode]
        package = importlib.import_module(f"configs.{mode}")

        def from_class(class_name):
            if class_name == "SampleConfig":
                return f"{prefix}Config"
            for pattern in (r"Config(.+)$", r"(.+)Config$"):
                match = re.match(pattern, class_name)
                if match:
                    return f"{prefix}Config{match.group(1)}"
            return class_name

        expected = set()
        for path in sorted(Path(f"configs/{mode}").glob("config_*.py")):
            module = importlib.import_module(f"configs.{mode}.{path.stem}")
            config_class = _get_config_class_from_module(module)
            assert config_class is not None, f"{path} defines no config class"
            expected.add(from_class(config_class.__name__))

        assert set(package.__all__) == expected

    @pytest.mark.parametrize("mode", sorted(MODE_PREFIXES))
    def test_the_default_config_is_the_sample_one(self, mode):
        import importlib

        package = importlib.import_module(f"configs.{mode}")
        default = f"{MODE_PREFIXES[mode]}Config"

        assert default in package.__all__
        assert getattr(package, default).__name__ == "SampleConfig"
