import importlib

PIPELINES = {
    "generate": "pipelines.generate",
    "hybrid": "pipelines.hybrid",
    "mosaic": "pipelines.mosaic",
    "scaling": "pipelines.scaling",
    "sweep": "pipelines.sweep",
    "compare": "pipelines.compare",
}


def load_pipeline(mode: str):
    module_path = PIPELINES.get(mode)
    if module_path is None:
        raise ValueError(
            f"Unknown MODE {mode!r}; a config must set one of {sorted(PIPELINES)}"
        )
    return importlib.import_module(module_path)
