# src/config/config.py
import logging
import os
from typing import Optional, Tuple

from .config_objects import FileConfig, ImageConfig, PlotConfig
from .enums import DataType

logger = logging.getLogger(__name__)


def load_idle():
    err_msg = "Logic of loading data is not implemented yet."
    logging.info(err_msg)
    raise NotImplementedError(err_msg)


class Config:
    """ Define base paths, this config file must be in the subdirectory of the project root"""

    SETS = []
    RESOLUTIONS = []

    DEFAULT_FRAME_SIZE: Tuple[int, int] = (510, 510)  # General (510, 510) # (width, height)
    CLOSED_NODES_FACTOR = 1.5
    CLOSED_EDGES_FACTOR = 1

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    MEASURE_WEIGHTED = False

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, 'input')
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'Original Graphs')

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .2  # Generally should be 0.15
    LOG_LEVEL = logging.INFO

    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')
    SYNTHETIC_GRAPH_DIRECTORY_NAME = 'synthetic_graphs'
    ORIGINAL_GRAPH_DIRECTORY_NAME = 'original_graphs'

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

    @staticmethod
    def _setup_logger(log_level=logging.INFO, details=""):
        _logger = logging.getLogger()
        if not _logger.hasHandlers():
            _logger.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            logs_dir = os.path.join(Config.BASE_OUTPUT_PATH, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, f'project_{details}.log'))
            file_handler.setLevel(log_level)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            _logger.addHandler(console_handler)
            _logger.addHandler(file_handler)

    @classmethod
    def initialize(cls):
        cls._setup_logger(cls.LOG_LEVEL)
        cls.POSITION_DATA_FUNC = load_idle
        cls.ADJ_MATRIX_DATA_FUNC = load_idle
        cls.IMAGES_FUNC = load_idle

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
            'frame_range': self.DEFAULT_FRAME_SIZE,
            'closed_nodes_factor': self.CLOSED_NODES_FACTOR,
            'closed_edges_factor': self.CLOSED_EDGES_FACTOR,
            'no_syn_graph': self.SYNTHETIC_GRAPH_NUMBER,
            'no_syn_network': self.SYNTHETIC_NETWORK_NUMBER,
            'error_tolerance': self.ERROR_TOLERANCE,
        }
        return f"Config: {_config})"

    @staticmethod
    def _find_file_with_pattern(directory_path, pattern, details='', must_exist=True) -> Optional[str]:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        files_in_directory = os.listdir(directory_path)
        matched_files = [file_name for file_name in files_in_directory if pattern.search(file_name)]

        if len(matched_files) == 1:
            file_path = os.path.join(directory_path, matched_files[0])
            logger.info(f"{details} file found: {file_path}")
            return file_path
        elif len(matched_files) > 1:
            raise FileExistsError(f"Multiple {details} files found: {matched_files}.")
        else:
            if must_exist:
                raise FileNotFoundError(f"No {details} file found in {directory_path}.")
            else:
                logger.warning(f"No {details} file found in {directory_path}.")
                return None
