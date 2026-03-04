# src/configs/sweep_mode/config_d.py
"""Sweep config for dataset D only (one machine)."""
from ..enums import DatasetId
from .config_sample import SampleConfig


class ConfigD(SampleConfig):
    DATASETS = [DatasetId("sample_D")]
