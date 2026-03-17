# src/configs/sweep_mode/config_b.py
"""Sweep config for dataset B only (one machine)."""
from ..enums import DatasetId
from .config_sample import SampleConfig


class ConfigB(SampleConfig):
    DATASETS = [DatasetId("sample_B")]
