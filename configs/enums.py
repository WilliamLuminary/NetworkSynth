import os
from enum import Enum, auto
from typing import Tuple


class DatasetId:
    """
    Flexible multi-level dataset identifier.

    Examples:
        DatasetId("1-0")              # Single level
        DatasetId("A", "10kX")        # Two levels
        DatasetId("exp1", "set_a", "high_res")  # Three levels

    Usage:
        dataset = DatasetId("1-0")
        str(dataset)          # "1-0"
        dataset.path          # "1-0"
        dataset.levels        # ("1-0",)
        dataset[0]            # "1-0"
        len(dataset)          # 1

        dataset = DatasetId("A", "10kX")
        str(dataset)          # "A/10kX"
        dataset.path          # "A/10kX" (or "A\\10kX" on Windows)
        dataset.levels        # ("A", "10kX")
        dataset[0]            # "A"
        dataset[1]            # "10kX"
    """

    __slots__ = ("_levels",)

    def __init__(self, *levels: str):
        if not levels:
            raise ValueError("DatasetId requires at least one level")
        # Filter out empty strings (like old IdleResolution.NA)
        filtered = tuple(lvl for lvl in levels if lvl)
        if not filtered:
            raise ValueError("DatasetId requires at least one non-empty level")
        object.__setattr__(self, "_levels", filtered)

    @property
    def levels(self) -> Tuple[str, ...]:
        return self._levels

    @property
    def path(self) -> str:
        return os.path.join(*self._levels)

    def __str__(self) -> str:
        return "/".join(self._levels)

    def __repr__(self) -> str:
        levels_str = ", ".join(f'"{lvl}"' for lvl in self._levels)
        return f"DatasetId({levels_str})"

    def __iter__(self):
        return iter(self._levels)

    def __len__(self) -> int:
        return len(self._levels)

    def __getitem__(self, index: int) -> str:
        return self._levels[index]

    def __hash__(self) -> int:
        return hash(self._levels)

    def __eq__(self, other) -> bool:
        if isinstance(other, DatasetId):
            return self._levels == other._levels
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, DatasetId):
            return self._levels < other._levels
        raise TypeError(f"Cannot compare DatasetId with {type(other)}")

    def __setattr__(self, name, value):
        raise AttributeError("DatasetId is immutable")


class AnalysisMode(Enum):
    BASIC = auto()
    FULL = auto()
    RETROACTIVE = auto()


class Mode(Enum):
    GEN = "generate_from_original_network"
    ANA = "multifractal_analyze"
    ATR = "generate_from_attributes"
