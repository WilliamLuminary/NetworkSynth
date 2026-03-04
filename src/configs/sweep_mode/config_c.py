# src/config/sweep_mode/config_c.py
"""Sweep config for dataset C only (one machine)."""
from ..enums import DatasetId
from .config_sample import SampleConfig


class ConfigC(SampleConfig):
    DATASETS = [DatasetId("sample_C")]
