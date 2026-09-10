# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
import sys
from pathlib import Path

import pytest

import networksynth
from networksynth.configs import BaseConfig, GenerateConfig, loader
from networksynth.configs.loader import config_in, load_config
from networksynth.pipelines import PIPELINES

_CONFIGS = Path(networksynth.__file__).resolve().parent / "configs"

pytestmark = pytest.mark.unit

_MINIMAL_GENERATE = """\
from networksynth.configs import DatasetId, GenerateConfig

CONFIG = GenerateConfig(
    DATASETS=[DatasetId("d")],
    FRAME_SIZE=(8, 8),
    SYNTHETIC_FRAME_SIZE=(8, 8),
    CLOSED_NODES_FACTOR=1.0,
    CLOSED_EDGES_FACTOR=1.0,
    MEASURE_WEIGHTED=False,
    {extra}
)
"""


def _config_modules_loaded():
    return {name for name in sys.modules if ".config_" in name}


def _fixture_package(root: Path, *names: str, extra: str = "") -> Path:
    """A configs package written for a test, so nothing asserts on a real one.

    load_config imports by dotted path relative to the package root and refuses
    anything outside it, so a fixture has to be a real importable module. The
    caller repoints that root at the tmp_path this returns.
    """
    package = root / "fixture_mode"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in names:
        (package / f"config_{name}.py").write_text(
            _MINIMAL_GENERATE.format(extra=extra)
        )
    return package


def _forget_fixture_modules():
    for name in [n for n in sys.modules if n.startswith("fixture_mode")]:
        del sys.modules[name]
    importlib.invalidate_caches()


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    """Each test gets its own package, so the import cache must not carry over."""
    monkeypatch.setattr(loader, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_fixture_modules()
    yield tmp_path
    _forget_fixture_modules()


class TestLoadingByPath:
    def test_it_returns_the_config_in_that_file(self, fixture_root):
        package = _fixture_package(fixture_root, "one", extra="SNAPSHOT_INTERVAL=10,")

        config = load_config(str(package / "config_one.py"))

        assert isinstance(config, GenerateConfig)
        assert config.SNAPSHOT_INTERVAL == 10

    def test_the_run_is_named_after_the_file(self, fixture_root):
        package = _fixture_package(fixture_root, "one")

        config = load_config(str(package / "config_one.py"))

        assert config.OUTPUT_DENOTE == "fixture_mode_config_one"

    def test_two_loads_are_two_runs(self, fixture_root):
        package = _fixture_package(fixture_root, "one")

        first = load_config(str(package / "config_one.py"))
        second = load_config(str(package / "config_one.py"))

        assert first.RUN_ID != second.RUN_ID

    def test_the_path_has_to_be_the_real_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(str(_CONFIGS / "hybrid_mode" / "config_snapshot"))

    def test_a_missing_file_says_so(self):
        with pytest.raises(FileNotFoundError, match="No config file at"):
            load_config(str(_CONFIGS / "generate_mode" / "config_nope.py"))

    def test_a_config_outside_the_project_is_refused(self, tmp_path):
        stray = tmp_path / "config_stray.py"
        stray.write_text(_MINIMAL_GENERATE.format(extra=""))

        with pytest.raises(ValueError, match="outside the project"):
            load_config(str(stray))


class TestOnlyWhatIsAskedForLoads:
    def test_importing_configs_imports_no_config_module(self):
        before = _config_modules_loaded()

        import networksynth.configs  # noqa:F401

        assert _config_modules_loaded() == before

    def test_loading_one_does_not_drag_in_the_others(self, fixture_root):
        package = _fixture_package(fixture_root, "one", "two")
        before = _config_modules_loaded()

        config = load_config(str(package / "config_one.py"))

        assert isinstance(config, GenerateConfig)
        newly_loaded = _config_modules_loaded() - before
        assert newly_loaded == {"fixture_mode.config_one"}, newly_loaded


class TestTheFileBindsCONFIG:
    def test_a_file_with_no_config_says_so(self):
        module = type(sys)("networksynth.configs.fake_mode.config_empty")

        with pytest.raises(AttributeError, match="defines no CONFIG"):
            config_in(module)

    def test_config_has_to_be_a_config(self):
        module = type(sys)("networksynth.configs.fake_mode.config_wrong")
        module.CONFIG = {"MODE": "generate"}

        with pytest.raises(AttributeError, match="defines no CONFIG"):
            config_in(module)


class TestEveryConfigInTheRepoLoads:
    """The one place that must use the real configs: this is what it asserts."""

    @pytest.mark.parametrize(
        "path", sorted(str(p) for p in _CONFIGS.glob("*_mode/config_*.py"))
    )
    def test_it_loads(self, path):
        config = load_config(path)

        assert isinstance(config, BaseConfig)
        assert config.MODE in PIPELINES
