# src/config/name_resolution_set.py
from enums import SetName, Resolution


class NameResolutionSet:
    def __init__(self, set_name: SetName, resolution: Resolution):
        self.set_name = set_name
        self.resolution = resolution

    def __repr__(self):
        return f"GraphSetConfig(set_name={self.set_name}, resolution={self.resolution})"

    def __str__(self):
        return f"Set name: {self.set_name}, Resolution: {self.resolution}"

    def __eq__(self, other):
        if not isinstance(other, NameResolutionSet):
            return False
        return self.set_name == other.set_name and self.resolution == other.resolution

    def __hash__(self):
        return hash((self.set_name, self.resolution))
