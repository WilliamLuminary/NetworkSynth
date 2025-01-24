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


FILE_CONFIGURATIONS = {
    DataType.ORIGINAL_IMAGE: ImageConfig(
        relative_dir="origin",
        data_type=DataType.ORIGINAL_IMAGE,
        alpha=0.6,
        detail="original_image"
    ),
    DataType.ORIGINAL_GRAPH: PlotConfig(
        relative_dir="origin",
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.ORIGINAL_GRAPH,
        detail="original_graph"
    ),
    DataType.ORIGINAL_NETWORK: FileConfig(
        relative_dir="origin",
        data_type=DataType.ORIGINAL_NETWORK,
        detail="original_network"
    ),
    DataType.SYNTHETIC_GRAPH: PlotConfig(
        relative_dir="synthetic",
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.SYNTHETIC_GRAPH,
        show_on_the_fly=False,
        detail="synthetic_graph",
    ),
    DataType.SYNTHETIC_NETWORK: FileConfig(
        relative_dir="synthetic",
        data_type=DataType.SYNTHETIC_NETWORK,
        detail="synthetic_network",
    ),
    DataType.DEFAULT_DATA: FileConfig(
        relative_dir="",
        data_type=DataType.DEFAULT_DATA,
    )
}
