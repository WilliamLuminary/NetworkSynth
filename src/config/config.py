# src/config/config.py
import logging
import os

from .enums import DataType
from .config_objects import FileConfig, ImageConfig, PlotConfig


class Config:
    FILE_CONFIGURATIONS = {
        DataType.ORIGINAL_IMAGE: ImageConfig(
            relative_dir="origin",
            data_type=DataType.ORIGINAL_IMAGE,
            alpha=0.6
        ),
        DataType.ORIGINAL_GRAPH: PlotConfig(
            relative_dir="origin",
            node_size=4.0,
            line_width=2.0,
            data_type=DataType.ORIGINAL_GRAPH,
        ),
        DataType.SYNTHETIC_GRAPH: PlotConfig(
            relative_dir="synthetic",
            node_size=4.0,
            line_width=2.0,
            data_type=DataType.SYNTHETIC_GRAPH,
        ),
        DataType.SYNTHETIC_NETWORK: FileConfig(
            relative_dir="synthetic",
            data_type=DataType.SYNTHETIC_NETWORK,
        )
    }

    # Define base paths, this config file must be in the subdirectory of the project root
    DEFAULT_FRAME_RANGE = 510
    CLOSED_NODES_FACTOR = 1
    CLOSED_EDGES_FACTOR = 1.5

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1
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
