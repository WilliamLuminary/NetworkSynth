# SPDX-License-Identifier: GPL-3.0-or-later
"""Worker processes must not each grab every core."""

import importlib
import inspect
import os

import pytest

pytestmark = pytest.mark.unit

_PIPELINES = (
    "networksynth.pipelines.generate",
    "networksynth.pipelines.hybrid",
    "networksynth.pipelines.mosaic",
    "networksynth.pipelines.sweep",
    "networksynth.pipelines.compare",
)


@pytest.mark.parametrize("module_name", _PIPELINES)
def test_pipeline_pins_omp_before_importing_anything_heavy(module_name):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    pin = 'os.environ.setdefault("OMP_NUM_THREADS", "1")'
    assert pin in source, f"{module_name} does not pin OMP_NUM_THREADS"

    lines = [line for line in source.splitlines() if line.strip()]
    position = next(i for i, line in enumerate(lines) if pin in line)
    heavy = ("numpy", "scipy", "matplotlib", "igraph", "cv2", "sklearn", "ot")
    too_early = [
        line
        for line in lines[:position]
        if line.startswith(("import ", "from "))
        and any(name in line.split() for name in heavy)
    ]
    assert not too_early, (
        f"{module_name} imports {too_early} before pinning OMP_NUM_THREADS; "
        "the pin only works if it happens before the maths libraries load"
    )


def test_the_pin_is_in_effect_once_a_pipeline_is_imported():
    importlib.import_module("networksynth.pipelines.generate")
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_nothing_imports_networkit_any_more():
    """igraph replaced it everywhere; this keeps it from creeping back."""
    roots = ("analysis", "configs", "graphs", "gui", "handlers", "pipelines", "utils")
    offenders = []
    for root in roots:
        for directory, _, names in os.walk(root):
            if "__pycache__" in directory:
                continue
            for name in names:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                with open(path) as handle:
                    if "networkit" in handle.read():
                        offenders.append(path)
    assert not offenders, f"still referencing networkit: {offenders}"
