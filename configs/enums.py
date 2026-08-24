import os
from enum import Enum, auto
from typing import Tuple


class DatasetId(tuple):
    """A dataset's identity: one or more path levels, innermost last.

    ``DatasetId("A", "10kX")`` names a dataset stored under ``A/10kX``.  A tuple
    subclass because that is what it is — an immutable sequence of levels — and
    inheriting gives equality, hashing, ordering, indexing and iteration
    without writing any of them.
    """

    def __new__(cls, *levels: str):
        filtered = tuple(level for level in levels if level)
        if not filtered:
            raise ValueError("DatasetId requires at least one non-empty level")
        return super().__new__(cls, filtered)

    def __getnewargs__(self) -> Tuple[str, ...]:
        # The levels *are* the constructor's arguments.  Without this, pickle
        # and copy rebuild a tuple subclass by handing the whole tuple to
        # __new__ as a single argument, which this variadic signature would
        # then nest one level deep.  A DatasetId crosses to every spawned
        # child, so getting this wrong breaks the run rather than a corner.
        return tuple(self)

    def __repr__(self) -> str:
        return f"DatasetId({', '.join(repr(level) for level in self)})"

    def __str__(self) -> str:
        return "/".join(self)

    @property
    def path(self) -> str:
        return os.path.join(*self)


class AnalysisMode(Enum):
    BASIC = auto()
    FULL = auto()
    RETROACTIVE = auto()
