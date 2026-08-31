"""Worker processes must not each grab every core.

This used to pin networkit's thread count. igraph is single-threaded, so what is
left to guard is the environment variable that stops numpy, scipy and OpenBLAS
from oversubscribing inside a pool worker.
"""

import importlib
import inspect
import os

import pytest

pytestmark = pytest.mark.unit

_PIPELINES = (
    "pipelines.generate",
    "pipelines.hybrid",
    "pipelines.mosaic",
    "pipelines.sweep",
    "pipelines.compare",
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
    importlib.import_module("pipelines.generate")
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_no_pipeline_still_talks_to_networkit():
    """The migration is done when this passes with networkit uninstalled."""
    offenders = []
    for module_name in _PIPELINES:
        source = inspect.getsource(importlib.import_module(module_name))
        if "networkit" in source or "nk." in source:
            offenders.append(module_name)
    assert not offenders, f"still referencing networkit: {offenders}"
