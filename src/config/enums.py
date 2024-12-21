# src/enums.py
from enum import Enum


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


class ResultType(Enum):
    ORIGINAL_IMAGE = "Original Image"
    ORIGINAL_GRAPH = "Original Graph"
    SYNTHETIC_GRAPH = "Synthetic Graph"
    SYNTHETIC_NETWORK = "Synthetic Network"

    def __str__(self):
        return self.value
