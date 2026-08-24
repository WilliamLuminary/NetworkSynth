import importlib
import inspect
from pathlib import Path
from typing import Type

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> Type:
    """Return the config class defined in the config file at *path*.

    The config is named by the file that holds it — ``configs/generate_mode/
    config_snapshot_1x1.py`` — so what you type is what is on disk, and ``ls``
    is the list of what you can run.  Nothing is imported but that file.

    Imported as a module of its package rather than loaded from the file
    directly, because a config module reaches its base class with a relative
    import (``from ..base_config import BaseConfig``), which only resolves
    inside the package.
    """
    module_path = Path(path).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"No config file at {path}")

    try:
        relative = module_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        raise ValueError(
            f"{path} is outside the project ({_PROJECT_ROOT}); a config has to "
            "live in the configs package so its relative imports resolve"
        ) from None

    module = importlib.import_module(".".join(relative.with_suffix("").parts))
    return config_class_in(module)


def config_class_in(module) -> Type:
    """The one config class *module* defines.

    A config file holds exactly one config: anything else makes "the config in
    this file" ambiguous, so say so rather than picking one.
    """
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and hasattr(obj, "DATASETS")
    ]
    if not candidates:
        raise AttributeError(f"{module.__name__} defines no config class")
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise AttributeError(f"{module.__name__} defines several configs: {names}")
    return candidates[0]
