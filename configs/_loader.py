import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Optional, Type

logger = logging.getLogger(__name__)


def _get_config_class_from_module(module) -> Optional[Type]:
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if not (name.endswith("Config") or name.startswith("Config")):
            continue
        if hasattr(obj, "DATASETS"):
            return obj
    return None


def _export_name(module_stem: str, mode_prefix: str) -> str:
    """The name a config module is exported under.

    Derived from the *filename*, not from the class inside it, so the index can
    be built without importing anything::

        config_sample.py       -> GenConfig          (the mode's default)
        config_nanowires.py    -> GenConfigNanowires
        config_snapshot_1x1.py -> GenConfigSnapshot1x1
    """
    rest = module_stem[len("config_") :]
    if rest == "sample":
        return f"{mode_prefix}Config"
    parts = (part[:1].upper() + part[1:] for part in rest.split("_"))
    return f"{mode_prefix}Config" + "".join(parts)


def index_configs(directory: Path, mode_prefix: str) -> Dict[str, str]:
    """Map export name -> module name for every config in *directory*.

    Nothing is imported: a config module pulls in cv2, csv readers and its own
    path arithmetic, and a run needs exactly one of them.  Importing all 17 to
    find the one being run cost about 200ms and ran module-level code for
    datasets that may not even exist on this machine.
    """
    return {
        _export_name(path.stem, mode_prefix): path.stem
        for path in sorted(directory.glob("config_*.py"))
    }


def load_config(package_name: str, module_stem: str, export_name: str) -> Type:
    """Import one config module and return its config class."""
    module = importlib.import_module(f"{package_name}.{module_stem}")
    config_class = _get_config_class_from_module(module)
    if config_class is None:
        raise AttributeError(
            f"{package_name}.{module_stem} defines no config class, so "
            f"{export_name!r} cannot be resolved"
        )
    return config_class
