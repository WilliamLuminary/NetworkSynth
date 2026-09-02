# SPDX-License-Identifier: GPL-3.0-or-later
import importlib

PIPELINES = {
    "generate": "networksynth.pipelines.generate",
    "hybrid": "networksynth.pipelines.hybrid",
    "mosaic": "networksynth.pipelines.mosaic",
    "scaling": "networksynth.pipelines.scaling",
    "sweep": "networksynth.pipelines.sweep",
    "compare": "networksynth.pipelines.compare",
}


def load_pipeline(mode: str):
    module_path = PIPELINES.get(mode)
    if module_path is None:
        raise ValueError(
            f"Unknown MODE {mode!r}; a config must set one of {sorted(PIPELINES)}"
        )
    return importlib.import_module(module_path)
