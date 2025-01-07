# src/config/config.py
import logging
import os

from .enums import DataType
from .config_objects import FileConfig, ImageConfig, PlotConfig


class Config:
    # Define base paths, this config file must be in the subdirectory of the project root
    DEFAULT_FRAME_RANGE = 510
    CLOSED_NODES_FACTOR = 1.5
    CLOSED_EDGES_FACTOR = 1
    # For 10_kx image, node fac should be 1.5, and edge fac should be 1

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15  # Generally should be 0.15

    LOG_LEVEL = logging.INFO

    FILE_CONFIGURATIONS = {
        DataType.ORIGINAL_IMAGE: ImageConfig(
            relative_dir="origin",
            data_type=DataType.ORIGINAL_IMAGE,
            alpha=0.6
        ),
        DataType.ORIGINAL_GRAPH: PlotConfig(
            relative_dir="origin",
            node_size=6.0,
            line_width=3.0,
            data_type=DataType.ORIGINAL_GRAPH,
        ),
        DataType.SYNTHETIC_GRAPH: PlotConfig(
            relative_dir="synthetic",
            node_size=6.0,
            line_width=3.0,
            data_type=DataType.SYNTHETIC_GRAPH,
            show_on_the_fly=False
        ),
        DataType.SYNTHETIC_NETWORK: FileConfig(
            relative_dir="synthetic",
            data_type=DataType.SYNTHETIC_NETWORK,
            detail=f"no_syn_nw_{SYNTHETIC_NETWORK_NUMBER}",
        ),
        DataType.DEFAULT_DATA: FileConfig(
            relative_dir="",
            data_type=DataType.DEFAULT_DATA,
        )
    }

    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

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
        cls._setup_logger(cls.LOG_LEVEL)

    @classmethod
    def set_node_factor(cls, factor: float):
        cls.CLOSED_NODES_FACTOR = factor
        logging.info(f"CLOSED_NODES_FACTOR has been overwritten! Current value: {factor}")

    @classmethod
    def set_edge_factor(cls, factor: float):
        cls.CLOSED_EDGES_FACTOR = factor
        logging.info(f"CLOSED_EDGES_FACTOR has been overwritten! Current value; {factor}")

    @classmethod
    def disable_saving(cls, reason: str = ""):
        Config.DISABLE_SAVING = True
        Config.DISABLE_SAVING_NOTE = reason

    @classmethod
    def enable_saving(cls, reason: str = ""):
        Config.DISABLE_SAVING = False
        Config.DISABLE_SAVING_NOTE = reason

    def __str__(self):
        _config = {
            'frame_range': self.DEFAULT_FRAME_RANGE,
            'closed_nodes_factor': self.CLOSED_NODES_FACTOR,
            'closed_edges_factor': self.CLOSED_EDGES_FACTOR,
            'no_syn_graph': self.SYNTHETIC_GRAPH_NUMBER,
            'no_syn_network': self.SYNTHETIC_NETWORK_NUMBER,
            'error_tolerance': self.ERROR_TOLERANCE,
        }
        return f"Config: {_config})"
