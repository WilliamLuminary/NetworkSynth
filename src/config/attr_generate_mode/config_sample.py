# src/config/attr_generate_mode/config_sample.py
import logging
import os
import pickle
from typing import Dict

from ..base_config import BaseConfig

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    IMAGE_SIZE = (1887, 2048)
    FRAME_SIZE = (1887 // 4, 2048 // 4)
    SYNTHETIC_FRAME_SIZE = FRAME_SIZE
    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = .8

    SYNTHETIC_GRAPH_NUMBER = 10
    SYNTHETIC_NETWORK_NUMBER = 300

    MAX_ATTEMPTS = 10

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'sample_input', 'generate_mode')
    ATTRIBUTES_DICT_DATA_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'sample_input', 'property_generate_mode')

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ATTRIBUTES_DICT_FUNC = cls._load_attr_dict
        cls._inject_dependencies()

    @staticmethod
    def _load_attr_dict(path) -> Dict:
        for file in os.listdir(path):
            if file.endswith('.pkl') and ('property' in file or 'attribute' in file):
                with open(os.path.join(path, file), 'rb') as f:
                    content = pickle.load(f)
                    return content
        return {}
