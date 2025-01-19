# src/config/enums.py
from enum import Enum
from typing import Set, Union


class SetName(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

    S4  = "W-2-89-1_004"
    S8  = "W-2-89-1_008"
    S11 = "W-2-89-1_011"
    S14 = "W-2-89-1_014"
    S17 = "W-2-89-1_017"
    S20 = "W-2-89-1_020"
    S23 = "W-2-89-1_023"
    S26 = "W-2-89-1_026"
    S29 = "W-2-89-1_029"
    S32 = "W-2-89-1_032"
    NA = ""

    def __str__(self):
        return self.value


class Resolution(Enum):
    X10K = "10kX"
    X15K = "15kX"
    X20K = "20kX"
    X30K = "30kX"
    NA = ""

    def __str__(self):
        return self.value


class FileType(Enum):
    PNG = "png"
    JPG = "jpg"
    PKL = "pkl"

    def __str__(self):
        return self.value


class FileTag(Enum):
    FIG = "figure"
    DATA = "data"
    PLOT = "plot"
    ORI = "original"
    SYN = "synthetic"

    def __str__(self):
        return self.value


class DataType(Enum):
    DEFAULT_DATA = ("Default Data", {FileTag.DATA}, FileType.PKL)
    ORIGINAL_IMAGE = ("Original Image", {FileTag.FIG, FileTag.ORI}, FileType.PNG)
    ORIGINAL_GRAPH = ("Original Graph", {FileTag.FIG, FileTag.PLOT, FileTag.ORI}, FileType.PNG)
    SYNTHETIC_GRAPH = ("Synthetic Graph", {FileTag.FIG, FileTag.PLOT, FileTag.SYN}, FileType.PNG)
    SYNTHETIC_NETWORK = ("Synthetic Network", {FileTag.DATA, FileTag.SYN}, FileType.PKL)

    def __init__(self, description: str, tags: Set[FileTag], file_extension: FileType):
        self.description = description
        self.tags = tags
        self.file_extension = file_extension

    def __str__(self):
        return self.description

    def has_tag(self, tags: Union[FileTag, tuple[FileTag], list[FileTag]]) -> bool:
        if isinstance(tags, FileTag):
            return tags in self.tags
        return set(tags).issubset(self.tags)
