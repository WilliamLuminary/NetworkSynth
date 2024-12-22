from dataclasses import dataclass
from typing import Optional

from config.enums import FileTag


@dataclass
class FileConfig:
    relative_dir: str
    file_tags: Optional[set[FileTag]]


@dataclass
class ImageConfig(FileConfig):
    detail_prefix: Optional[str]


@dataclass
class PlotConfig(ImageConfig):
    node_size: float
    line_width: float
