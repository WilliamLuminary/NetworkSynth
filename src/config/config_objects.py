# src/config/config_objects.py
from dataclasses import dataclass
from typing import Optional

from .enums import DataType, Resolution, SetName


@dataclass
class FileConfig:
    relative_dir: str
    detail: Optional[str]
    data_type: DataType

    def __post_init__(self):
        self.file_tags = self.data_type.tags
        self.file_extension = self.data_type.file_extension


@dataclass
class ImageConfig(FileConfig):
    alpha: Optional[float]


@dataclass
class PlotConfig(FileConfig):
    node_size: float
    line_width: float


@dataclass(frozen=True)
class NameResolutionSet:
    set_name: SetName
    resolution: Resolution

    def __str__(self):
        return f"Set name: {self.set_name}, Resolution: {self.resolution}"
