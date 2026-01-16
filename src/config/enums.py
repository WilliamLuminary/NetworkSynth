# src/config/enums.py
from enum import Enum, auto
from typing import Set, Union


class SetName(Enum):
    """Base class for all config-specific set enums."""

    def __str__(self):
        return self.value

    def __lt__(self, other):
        if type(self) is not type(other):
            raise TypeError("Comparisons between different Set types are not allowed.")
        return self.value < other.value


class Resolution(Enum):
    """Base class for all config-specific resolution enums."""

    def __str__(self):
        return self.value

    def __lt__(self, other):
        if type(self) is not type(other):
            raise TypeError("Comparisons between different Set types are not allowed.")
        return self.value < other.value


class IdleResolution(Resolution):
    """Default resolution used when no specific resolution is needed."""

    NA = ""


class FileExtension(Enum):
    PNG = "png"
    PKL = "pkl"

    def __str__(self):
        return self.value


class FileTag(Enum):
    FIG = "figure"
    PLOT = "plot"
    DATA = "data"

    ORI = "original"
    SYN = "synthetic"
    ANA = "analysis"

    def __str__(self):
        return self.value


class DataType(Enum):
    DEFAULT_DATA = ("", {FileTag.DATA}, FileExtension.PKL)
    ORIGINAL_IMAGE = ("Original Image", {FileTag.FIG, FileTag.ORI}, FileExtension.PNG)
    ORIGINAL_GRAPH = (
        "Original Graph",
        {FileTag.FIG, FileTag.PLOT, FileTag.ORI},
        FileExtension.PNG,
    )
    ORIGINAL_PROPERTY = (
        "Original Property",
        {FileTag.ORI, FileTag.DATA},
        FileExtension.PKL,
    )
    ORIGINAL_NETWORK = (
        "Original Network",
        {FileTag.DATA, FileTag.ORI},
        FileExtension.PKL,
    )
    SYNTHETIC_GRAPH = (
        "Synthetic Graph",
        {FileTag.FIG, FileTag.PLOT, FileTag.SYN},
        FileExtension.PNG,
    )
    SYNTHETIC_NETWORK = (
        "Synthetic Network",
        {FileTag.DATA, FileTag.SYN},
        FileExtension.PKL,
    )
    ANALYSIS_DATA = ("Analysis Data", {FileTag.DATA, FileTag.ANA}, FileExtension.PKL)
    ANALYSIS_FIGURE = (
        "Analysis Figure",
        {FileTag.FIG, FileTag.PLOT, FileTag.ANA},
        FileExtension.PNG,
    )

    def __init__(
        self, description: str, tags: Set[FileTag], file_extension: FileExtension
    ):
        self.description = description
        self.tags = tags
        self.file_extension = file_extension

    def __str__(self):
        return self.description

    def has_tag(self, tags: Union[FileTag, tuple[FileTag], list[FileTag]]) -> bool:
        if isinstance(tags, FileTag):
            return tags in self.tags
        return set(tags).issubset(self.tags)


class AnalysisMode(Enum):
    BASIC = auto()
    FULL = auto()
    RETROACTIVE = auto()


class Mode(Enum):
    GEN = "generate_from_original_network"
    ANA = "multifractal_analyze"
    ATR = "generate_from_attributes"
