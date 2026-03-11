# src/configs/sweep_mode/config_a.py
"""Sweep config for dataset A only (one machine)."""
from ..enums import DatasetId
from .config_sample import SampleConfig


class ConfigA(SampleConfig):
    DATASETS = [DatasetId("sample_A")]
