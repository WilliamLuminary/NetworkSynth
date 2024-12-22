# src/enums.py
from enum import Enum
from typing import Set, Union


class SetName(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

    def __str__(self):
        return self.value


class Resolution(Enum):
    X10K = "10kX"
    X15K = "15kX"
    X20K = "20kX"
    X30K = "30kX"

    def __str__(self):
        return self.value


class FileTag(Enum):
    FIG = "figure"
    PKL = "pickle"
    PLOT = "plot"
    ORI = "original"
    SYN = "synthetic"

    def __str__(self):
        return self.value


class DataType(Enum):
    ORIGINAL_IMAGE = ("Original Image", {FileTag.FIG, FileTag.ORI})
    ORIGINAL_GRAPH = ("Original Graph", {FileTag.FIG, FileTag.PLOT, FileTag.ORI})
    SYNTHETIC_GRAPH = ("Synthetic Graph", {FileTag.FIG, FileTag.PLOT, FileTag.SYN})
    SYNTHETIC_NETWORK = ("Synthetic Network", {FileTag.PKL, FileTag.SYN})

    def __init__(self, description: str, tags: Set[FileTag]):
        self.description = description
        self.tags = tags

    def __str__(self):
        return self.description

    def has_tag(self, tags: Union[FileTag, tuple[FileTag], list[FileTag]]) -> bool:
        if isinstance(tags, FileTag):
            return tags in self.tags
        return set(tags).issubset(self.tags)
