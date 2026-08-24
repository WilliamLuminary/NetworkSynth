"""The pipelines, and the config field that selects one.

A config says which pipeline runs it (``MODE``); this maps that to the module.
Both entry points read this one table — ``run.py`` for a config on disk and
``gui_run.py`` for a run-spec — so a new pipeline is registered once.
"""

import importlib

#: MODE -> module path.  The keys are the vocabulary a config's MODE draws from.
PIPELINES = {
    "generate": "pipelines.generate",
    "hybrid": "pipelines.hybrid",
    "mosaic": "pipelines.mosaic",
    "scaling": "pipelines.scaling",
    "sweep": "pipelines.sweep",
    "compare": "pipelines.compare",
}


def load_pipeline(mode: str):
    """Import and return the pipeline module for *mode*.

    Imported here rather than at module import, because a pipeline pulls in
    networkit and sets OMP threads: a run wants one of them, not six.
    """
    module_path = PIPELINES.get(mode)
    if module_path is None:
        raise ValueError(
            f"Unknown MODE {mode!r}; a config must set one of {sorted(PIPELINES)}"
        )
    return importlib.import_module(module_path)
