# SPDX-License-Identifier: GPL-3.0-or-later
import json

import pytest

from networksynth.handlers.manifest import STATUS_CANCELLED, STATUS_FAILED, STATUS_OK

pytestmark = pytest.mark.unit


CHECKED_PIPELINES = [
    ("networksynth.pipelines.generate", "run_for_dataset"),
    ("networksynth.pipelines.hybrid", "_run_dataset_in_subprocess"),
    ("networksynth.pipelines.sweep", "run_for_dataset"),
    ("networksynth.pipelines.compare", "run_for_dataset"),
]


def _prepare(module, runner, replacement, monkeypatch):
    monkeypatch.setattr(module, runner, replacement)


def _config(tmp_path, name):
    from dataclasses import replace

    from tests.fixture_config import FIXTURES

    mode = name.rsplit(".", 1)[-1]
    return replace(
        FIXTURES[mode],
        BASE_OUTPUT_PATH=str(tmp_path / f"out_{name}"),
        DISABLE_SAVING=False,
    )


def _read_manifest(tmp_path, name):
    import os

    root = tmp_path / f"out_{name}"
    found = {os.path.realpath(p): p for p in root.glob("*/manifest.json")}
    assert len(found) == 1, f"expected exactly one manifest, got {sorted(found)}"
    return json.loads(next(iter(found.values())).read_text())


@pytest.mark.parametrize("module_path,runner", CHECKED_PIPELINES)
def test_a_completed_run_writes_a_manifest(module_path, runner, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(module_path)
    _prepare(module, runner, lambda *a, **k: None, monkeypatch)

    module.main(config=_config(tmp_path, module_path))

    assert _read_manifest(tmp_path, module_path)["status"] == STATUS_OK


@pytest.mark.parametrize("module_path,runner", CHECKED_PIPELINES)
def test_a_failed_run_writes_a_manifest_saying_so(
    module_path, runner, tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module(module_path)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    _prepare(module, runner, explode, monkeypatch)

    with pytest.raises(RuntimeError):
        module.main(config=_config(tmp_path, module_path))

    manifest = _read_manifest(tmp_path, module_path)
    assert manifest["status"] == STATUS_FAILED
    assert "boom" in manifest["error"]


@pytest.mark.parametrize("module_path,runner", CHECKED_PIPELINES)
def test_a_cancelled_run_is_distinct_from_a_failed_one(
    module_path, runner, tmp_path, monkeypatch
):
    import importlib

    module = importlib.import_module(module_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    _prepare(module, runner, interrupt, monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        module.main(config=_config(tmp_path, module_path))

    assert _read_manifest(tmp_path, module_path)["status"] == STATUS_CANCELLED


def test_every_gui_mode_is_covered_here():
    from networksynth import gui_run
    from networksynth.pipelines import PIPELINES

    checked = {path for path, _ in CHECKED_PIPELINES}
    offered = {PIPELINES[mode] for mode in gui_run._GUI_MODES}

    assert offered <= checked, f"unchecked GUI modes: {offered - checked}"
