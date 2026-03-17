"""Dynamic config loader for *_mode directories."""

import importlib
import inspect
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Type

logger = logging.getLogger(__name__)


def _get_config_class_from_module(module) -> Optional[Type]:
    """Find the first config class defined in the module."""
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if not (name.endswith("Config") or name.startswith("Config")):
            continue
        if hasattr(obj, "DATASETS"):
            return obj
    return None


def _derive_export_name(class_name: str, mode_prefix: str) -> str:
    """
    Derive export name based on class name and mode prefix.

    Naming convention (mode_prefix="Gen"):
        - SampleConfig -> GenConfig (default)
        - ConfigXxx -> GenConfigXxx
    """
    if class_name == "SampleConfig":
        return f"{mode_prefix}Config"

    config_match = re.match(r"Config(.+)$", class_name)
    if config_match:
        return f"{mode_prefix}Config{config_match.group(1)}"

    suffix_match = re.match(r"(.+)Config$", class_name)
    if suffix_match:
        return f"{mode_prefix}Config{suffix_match.group(1)}"

    return class_name


def load_configs_from_directory(
    directory: Path,
    package_name: str,
    mode_prefix: str,
) -> Dict[str, Type]:
    """Dynamically load all config classes from config_*.py files."""
    configs = {}
    config_files = sorted(directory.glob("config_*.py"))

    for config_file in config_files:
        module_name = config_file.stem
        full_module_name = f"{package_name}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
            config_class = _get_config_class_from_module(module)
            if config_class is None:
                continue

            class_name = config_class.__name__
            export_name = _derive_export_name(class_name, mode_prefix)
            configs[export_name] = config_class

        except (ImportError, AttributeError) as e:
            logger.warning("Failed to load config from %s: %s", config_file, e)

    return configs


def get_all_config_names(configs: Dict[str, Type]) -> list:
    return list(configs.keys())
