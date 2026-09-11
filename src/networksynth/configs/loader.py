# SPDX-License-Identifier: GPL-3.0-or-later
import importlib
import uuid
from dataclasses import replace
from pathlib import Path

from .base_config import BaseConfig

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str) -> BaseConfig:
    module_path = Path(path).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"No config file at {path}")

    try:
        relative = module_path.relative_to(_PACKAGE_ROOT)
    except ValueError:
        raise ValueError(
            f"{path} is outside the project ({_PACKAGE_ROOT}); a config has to "
            "live in the configs package so its relative imports resolve"
        ) from None

    parts = relative.with_suffix("").parts
    config = config_in(importlib.import_module(".".join(parts)))
    return replace(
        config, OUTPUT_DENOTE="_".join(parts[-2:]), RUN_ID=uuid.uuid4().hex[:8]
    )


def config_in(module) -> BaseConfig:
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, BaseConfig):
        raise AttributeError(
            f"{module.__name__} defines no CONFIG; a config file binds that name "
            "to a GenerateConfig, HybridConfig, SweepConfig or CompareConfig"
        )
    return config
