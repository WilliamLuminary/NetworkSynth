# src/config/file_definitions.py
from dataclasses import dataclass
from typing import Optional

from .enums import DataType, Resolution, SetName


@dataclass
class FileConfig:
    relative_dir: str
    data_type: DataType
    detail: Optional[str] = None

    def __post_init__(self):
        self.file_tags = self.data_type.tags
        self.file_extension = self.data_type.file_extension


@dataclass
class ImageConfig(FileConfig):
    alpha: Optional[float] = 0.6


@dataclass
class PlotConfig(FileConfig):
    node_size: float = 6.0
    line_width: float = 3.0
    show_on_the_fly: bool = True


@dataclass(frozen=True)
class NameResolutionSet:
    set_name: SetName
    resolution: Resolution

    def __str__(self):
        return f"Set name: {self.set_name}, Resolution: {self.resolution}"
