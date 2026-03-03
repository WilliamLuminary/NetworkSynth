"""
Analyze mode configurations - dynamically loads all config_*.py files.

Naming convention:
    config_sample.py (SampleConfig) -> AnaConfig
    config_xxx.py (ConfigXxx) -> AnaConfigXxx
"""

from pathlib import Path

from .._loader import get_all_config_names, load_configs_from_directory

_configs = load_configs_from_directory(
    directory=Path(__file__).parent,
    package_name=__name__,
    mode_prefix="Ana",
)

globals().update(_configs)
__all__ = get_all_config_names(_configs)
