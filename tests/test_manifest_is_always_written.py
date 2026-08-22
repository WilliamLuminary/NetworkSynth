import json

import pytest

from handlers.manifest import STATUS_CANCELLED, STATUS_FAILED, STATUS_OK

pytestmark = pytest.mark.unit


#: (module path, the per-dataset function main() calls)
PIPELINES = [
    ("pipelines.generate", "run_for_dataset"),
    ("pipelines.mosaic", "run_mosaic_for_dataset"),
    ("pipelines.scaling", "run_scaling_for_dataset"),
    ("pipelines.hybrid", "_run_dataset_in_subprocess"),
    ("pipelines.sweep", "run_for_dataset"),
    ("pipelines.analyze", "run_for_container"),
]


class _NoWandb:
    """Stands in for the wandb module so a sweep never reaches the network."""

    def login(self, *args, **kwargs):
        return True


def _prepare(module, runner, replacement, monkeypatch):
    """Patch the per-dataset call, plus whatever else main() would reach.

    A sweep would otherwise try to authenticate, and analyse would find no
    result directories to work through.
    """
    monkeypatch.setattr(module, runner, replacement)
    if hasattr(module, "wandb"):
        monkeypatch.setattr(module, "wandb", _NoWandb())
    if hasattr(module, "find_pkl_containers"):
        monkeypatch.setattr(module, "find_pkl_containers", lambda *a, **k: {"": "in"})


def _config(tmp_path, name):
    from configs import BaseConfig, DatasetId

    class Config(BaseConfig):
        BASE_OUTPUT_PATH = str(tmp_path / f"out_{name}")
        DATASETS = [DatasetId("ds")]
        SYNTHETIC_NETWORK_NUMBER = 1  # a sweep refuses to run without any

    return Config


def _read_manifest(tmp_path, name):
    import os

    root = tmp_path / f"out_{name}"
    found = {os.path.realpath(p): p for p in root.glob("*/manifest.json")}
    assert len(found) == 1, f"expected exactly one manifest, got {sorted(found)}"
    return json.loads(next(iter(found.values())).read_text())


@pytest.mark.parametrize("module_path,runner", PIPELINES)
def test_a_completed_run_writes_a_manifest(module_path, runner, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(module_path)
    _prepare(module, runner, lambda *a, **k: None, monkeypatch)

    module.main(config_cls=_config(tmp_path, module_path))

    assert _read_manifest(tmp_path, module_path)["status"] == STATUS_OK


@pytest.mark.parametrize("module_path,runner", PIPELINES)
def test_a_failed_run_writes_a_manifest_saying_so(
    module_path, runner, tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module(module_path)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    _prepare(module, runner, explode, monkeypatch)

    with pytest.raises(RuntimeError):
        module.main(config_cls=_config(tmp_path, module_path))

    manifest = _read_manifest(tmp_path, module_path)
    assert manifest["status"] == STATUS_FAILED
    assert "boom" in manifest["error"]


@pytest.mark.parametrize("module_path,runner", PIPELINES)
def test_a_cancelled_run_is_distinct_from_a_failed_one(
    module_path, runner, tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module(module_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    _prepare(module, runner, interrupt, monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        module.main(config_cls=_config(tmp_path, module_path))

    assert _read_manifest(tmp_path, module_path)["status"] == STATUS_CANCELLED


def test_every_gui_mode_is_covered_here():
    import gui_run

    checked = {path for path, _ in PIPELINES}
    offered = set(gui_run._MODES.values())

    assert offered <= checked, f"unchecked GUI modes: {offered - checked}"
