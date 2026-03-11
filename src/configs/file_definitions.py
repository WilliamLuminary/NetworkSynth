# src/configs/file_definitions.py
from dataclasses import dataclass
from typing import Optional

from .enums import DataType


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


ORIGINAL_DIR = "original"
SYNTHETIC_DIR = "synthetic"
INPLACE_DIR = ""
FILE_CONFIGURATIONS = {
    DataType.ORIGINAL_IMAGE: ImageConfig(
        relative_dir=ORIGINAL_DIR,
        data_type=DataType.ORIGINAL_IMAGE,
        alpha=0.6,
        detail="original_image",
    ),
    DataType.ORIGINAL_GRAPH: PlotConfig(
        relative_dir=ORIGINAL_DIR,
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.ORIGINAL_GRAPH,
        detail="original_graph",
    ),
    DataType.ORIGINAL_NETWORK: FileConfig(
        relative_dir=ORIGINAL_DIR,
        data_type=DataType.ORIGINAL_NETWORK,
        detail="original_network",
    ),
    DataType.ORIGINAL_PROPERTY: FileConfig(
        relative_dir=ORIGINAL_DIR,
        data_type=DataType.ORIGINAL_PROPERTY,
        detail="original_property",
    ),
    DataType.SYNTHETIC_GRAPH: PlotConfig(
        relative_dir=SYNTHETIC_DIR,
        node_size=6.0,
        line_width=3.0,
        data_type=DataType.SYNTHETIC_GRAPH,
        show_on_the_fly=False,
        detail="synthetic_graph",
    ),
    DataType.SYNTHETIC_GRAPH_PNG: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_GRAPH_PNG,
        detail="synthetic_graph_png",
    ),
    DataType.SYNTHETIC_NETWORK: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_NETWORK,
        detail="synthetic_network",
    ),
    DataType.SYNTHETIC_EDGELIST: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_EDGELIST,
        detail="synthetic_edgelist",
    ),
    DataType.SYNTHETIC_POSITIONS: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_POSITIONS,
        detail="synthetic_positions",
    ),
    DataType.SYNTHETIC_NETWORK_NKI: FileConfig(
        relative_dir=SYNTHETIC_DIR,
        data_type=DataType.SYNTHETIC_NETWORK_NKI,
        detail="synthetic_network_nki",
    ),
    DataType.ANALYSIS_DATA: FileConfig(
        relative_dir=INPLACE_DIR,
        data_type=DataType.ANALYSIS_DATA,
        detail="analysis_data",
    ),
    DataType.ANALYSIS_FIGURE: PlotConfig(
        relative_dir=INPLACE_DIR,
        data_type=DataType.ANALYSIS_FIGURE,
        detail="analysis_figure",
    ),
    DataType.DEFAULT_DATA: FileConfig(
        relative_dir="",
        data_type=DataType.DEFAULT_DATA,
    ),
}
