import os
import logging

from config.base import BaseConfig


class Config(BaseConfig):
    def __init__(self, set_name, resolution):
        super().__init__()
        self.set_name = set_name
        self.resolution = resolution

        self.SYNTHETIC_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, set_name, self.SYNTHETIC_GRAPH_DIRECTORY_NAME)
        self.ORIGINAL_GRAPH_PATH = os.path.join(self.OUTPUT_DIR, resolution, self.ORIGINAL_GRAPH_DIRECTORY_NAME)

