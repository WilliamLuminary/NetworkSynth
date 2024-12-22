# src/utils/config.py
import logging
import os

from enums import DataType
from plot_config import *


class Config:
    FILE_CONFIGURATIONS = {
        DataType.ORIGINAL_IMAGE: ImageConfig(
            relative_dir="origin",
            file_tags=DataType.ORIGINAL_IMAGE.tags,
            detail_prefix=None
        ),
        DataType.ORIGINAL_GRAPH: PlotConfig(
            relative_dir="origin",
            node_size=3.0,
            line_width=1.5,
            file_tags=DataType.ORIGINAL_GRAPH.tags,
            detail_prefix=None
        ),
        DataType.SYNTHETIC_GRAPH: PlotConfig(
            relative_dir="synthetic",
            node_size=3.0,
            line_width=1.5,
            file_tags=DataType.SYNTHETIC_GRAPH.tags,
            detail_prefix=None
        ),
        DataType.SYNTHETIC_NETWORK: FileConfig(
            relative_dir="synthetic",
            file_tags=DataType.SYNTHETIC_NETWORK.tags,
        )
    }

    # Define base paths, this config file must be in the subdirectory of the project root
    DEFAULT_FRAME_RANGE = 510
    CLOSED_NODES_FACTOR = 1
    CLOSED_EDGES_FACTOR = 1.5

    SYNTHETIC_GRAPH_NUMBER = 10
    SYNTHETIC_NETWORK_NUMBER = 300
    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, 'input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_PATH, 'results')

    SYNTHETIC_GRAPH_DIRECTORY_NAME = 'synthetic_graphs'
    ORIGINAL_GRAPH_DIRECTORY_NAME = 'original_graphs'

    @staticmethod
    def _setup_logger(log_level=logging.INFO):
        logger = logging.getLogger()
        if not logger.hasHandlers():
            logger.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            logs_dir = os.path.join(Config.BASE_OUTPUT_PATH, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, 'project.log'))
            file_handler.setLevel(log_level)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

    @classmethod
    def initialize(cls):
        cls._setup_logger()
