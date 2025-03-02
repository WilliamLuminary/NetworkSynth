# src/config/base_config.py
import inspect
import logging
import os
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_idle(*_, **__):
    err_msg = "Logic of loading data is not implemented yet."
    logging.info(err_msg)
    raise NotImplementedError(err_msg)


class BaseConfig:
    """ Define base paths, this config file must be in the subdirectory of the project root"""

    SETS: Union[list, np.ndarray]
    RESOLUTIONS = Union[list, np.ndarray]

    TRIM_SIZE: Tuple[int, int] = (0, 0, 0, 0)
    DEFAULT_FRAME_SIZE: Tuple[int, int] = None

    CLOSED_NODES_FACTOR: float
    CLOSED_EDGES_FACTOR: float

    MEASURE_WEIGHTED: bool
    FULL_ANALYSIS: bool = False

    SYNTHETIC_GRAPH_NUMBER: int = 0
    SYNTHETIC_NETWORK_NUMBER: int = 0

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data')

    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, 'input')

    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'position_data')
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, 'sparse_matrices_data')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'original_images_data')
    NETWORKS_DATA_PATH = None
    ATTRIBUTES_DICT_DATA_PATH = None

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15  # Generally should be 0.15

    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')
    POSITION_DATA_FUNC = ADJ_MATRIX_DATA_FUNC = IMAGES_FUNC = load_idle
    NETWORKS_FUNC = load_idle
    ATTRIBUTES_DICT_FUNC = load_idle

    @classmethod
    def _setup_logger(cls, log_level=None, details=None):
        log_level = log_level or logging.INFO
        details = details or ""
        _logger = logging.getLogger()
        if not _logger.hasHandlers():
            _logger.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            logs_dir = os.path.join(BaseConfig.BASE_OUTPUT_PATH, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, f'project_{details}.log'))
            file_handler.setLevel(log_level)
            # noinspection SpellCheckingInspection
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            _logger.addHandler(console_handler)
            _logger.addHandler(file_handler)

    @classmethod
    def initialize(cls):
        cls._setup_logger(details=cls.__name__)
        cls.OUTPUT_DENOTE = cls.__name__

    @classmethod
    def _update_attrs_in_base_config(cls):
        for name in dir(cls):
            if name.isupper() and not name.startswith('__'):
                value = getattr(cls, name)
                setattr(BaseConfig, name, value)

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
        BaseConfig.DISABLE_SAVING = True
        BaseConfig.DISABLE_SAVING_NOTE = reason

    @classmethod
    def enable_saving(cls, reason: str = ""):
        BaseConfig.DISABLE_SAVING = False
        BaseConfig.DISABLE_SAVING_NOTE = reason

    @classmethod
    def update_frame_size(cls, frame_size: Tuple[int, int]):
        cls.DEFAULT_FRAME_SIZE = frame_size
        logging.info(f"DEFAULT_FRAME_SIZE has been overwritten! Current value: {frame_size}")

    def __str__(self):
        def is_method_like(attr_value):
            return (inspect.isfunction(attr_value)
                    or inspect.ismethod(attr_value)
                    or isinstance(attr_value, staticmethod)
                    or isinstance(attr_value, classmethod))

        _class_vars = {
            k: v
            for k, v in vars(self.__class__).items()
            if not k.startswith("__") and not is_method_like(v)
        }
        _instance_vars = {k: v for k, v in vars(self).items() if not k.startswith('__') and not callable(v)}
        _config_dict = {**_class_vars, **_instance_vars}
        return f"{self.__class__.__name__}: {_config_dict}"

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
