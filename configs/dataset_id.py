import os
from typing import Tuple


class DatasetId(tuple):

    def __new__(cls, *levels: str):
        filtered = tuple(level for level in levels if level)
        if not filtered:
            raise ValueError("DatasetId requires at least one non-empty level")
        return super().__new__(cls, filtered)

    def __getnewargs__(self) -> Tuple[str, ...]:
        return tuple(self)

    def __repr__(self) -> str:
        return f"DatasetId({', '.join(repr(level) for level in self)})"

    def __str__(self) -> str:
        return "/".join(self)

    @property
    def path(self) -> str:
        return os.path.join(*self)
