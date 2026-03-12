# src/configs/base_config.py
import logging
import os
from typing import List, Optional, Tuple

from .enums import DatasetId

logger = logging.getLogger(__name__)


def load_idle(*_, **__):
    err_msg = "Logic of loading data is not implemented yet."
    logging.info(err_msg)
    raise NotImplementedError(err_msg)


class BaseConfig:
    """Define base paths, this config file must be in the subdirectory of the project root"""

    DATASETS: Optional[List[DatasetId]] = None

    @classmethod
    def get_datasets(cls) -> List[DatasetId]:
        """Get all datasets to process."""
        if cls.DATASETS is None:
            raise ValueError("DATASETS must be defined in the config")
        return cls.DATASETS

    # IMAGE_SIZE: Background image dimensions (height, width) in pixels
    IMAGE_SIZE: Tuple[int, int] = None

    # FRAME_SIZE: Original network plotting frame (X_range, Y_range)
    FRAME_SIZE: Tuple[int, int] = None

    # SYNTHETIC_FRAME_SIZE: Synthetic network generation frame (X_range, Y_range)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = None

    CLOSED_NODES_FACTOR: float
    CLOSED_EDGES_FACTOR: float

    MEASURE_WEIGHTED: bool
    FULL_ANALYSIS: bool = False
    FULL_Q_BAND: bool = False

    SYNTHETIC_GRAPH_NUMBER: int = 0
    SYNTHETIC_NETWORK_NUMBER: int = 0

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, "input")

    NETWORKS_DATA_PATH = None
    ATTRIBUTES_DICT_DATA_PATH = None

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = 0.15  # Generally should be 0.15

    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "output")
    ORIGINAL_NETWORK_FUNC = ORIGINAL_IMAGE_FUNC = load_idle
    NETWORKS_FUNC = load_idle
    ATTRIBUTES_DICT_FUNC = load_idle

    MAX_WORKERS: int = 50

    @classmethod
    def get_max_workers(cls, num_tasks: int = None) -> int:
        """Return number of worker processes, capped by MAX_WORKERS."""
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(1, cpu_count - 1), cls.MAX_WORKERS)
        if num_tasks is not None:
            max_workers = min(max_workers, num_tasks)
        return max_workers

    @classmethod
    def initialize(cls):
        cls._setup_logger(details=cls.__name__)
        module_parts = cls.__module__.split(".")
        mode = next((p for p in module_parts if p.endswith("_mode")), None)
        cls.OUTPUT_DENOTE = f"{mode}_{cls.__name__}" if mode else cls.__name__

    @classmethod
    def save(cls, identifier: str):
        """Return the save specs for *identifier*.

        Looks up ``save_<identifier>`` on the active config and calls it.
        Each ``save_*`` classmethod returns a list of tuples::

            (relative_dir, detail, extension, save_fn)

        An optional 5th element ``False`` disables the timestamp.
        """
        method = getattr(cls, f"save_{identifier}", None)
        if method is None:
            raise ValueError(
                f"No save method 'save_{identifier}' defined in {cls.__name__}."
            )
        return method()

    # ------------------------------------------------------------------
    # Default save methods (generate-mode baseline).
    # Mode configs override only what they need to change.
    # Each returns a list of (relative_dir, detail, ext, save_fn[, use_ts]).
    # ------------------------------------------------------------------

    @classmethod
    def save_original_image(cls):
        from .file_definitions import save_png

        return [("original", "original_image", "png", save_png)]

    @classmethod
    def save_original_network(cls):
        from .file_definitions import save_network_csv, save_network_nkbin

        return [
            ("original", "original_network", "csv", save_network_csv),
            ("original", "original_network", "nkbin", save_network_nkbin),
        ]

    @classmethod
    def save_original_property(cls):
        from .file_definitions import save_pickle

        return [("original", "original_property", "pkl", save_pickle)]

    @classmethod
    def save_original_graph(cls):
        from .file_definitions import save_svg

        return [("original", "original_graph", "svg", save_svg)]

    @classmethod
    def save_synthetic_graph(cls):
        from .file_definitions import save_webp

        return [("synthetic", "synthetic_graph", "webp", save_webp)]

    @classmethod
    def save_synthetic_network(cls):
        from .file_definitions import save_pickle

        return [("synthetic", "synthetic_network", "pkl", save_pickle)]

    @classmethod
    def save_synthetic_export(cls):
        from .file_definitions import save_network_csv, save_network_nkbin

        return [
            ("synthetic", "synthetic_network", "csv", save_network_csv),
            ("synthetic", "synthetic_network", "nkbin", save_network_nkbin),
        ]

    @classmethod
    def save_analysis_data(cls):
        from .file_definitions import save_pickle

        return [("", "analysis_data", "pkl", save_pickle, False)]

    @classmethod
    def save_analysis_figure(cls):
        from .file_definitions import save_svg

        return [("", "analysis_figure", "svg", save_svg)]

    @classmethod
    def _inject_dependencies(cls):
        for name in dir(cls):
            if name.startswith("__"):
                continue
            if name.isupper() or name.startswith("save_"):
                value = getattr(cls, name)
                setattr(BaseConfig, name, value)

    @classmethod
    def set_node_factor(cls, factor: float):
        cls.CLOSED_NODES_FACTOR = factor
        logging.info(
            f"CLOSED_NODES_FACTOR has been overwritten! Current value: {factor}"
        )

    @classmethod
    def set_edge_factor(cls, factor: float):
        cls.CLOSED_EDGES_FACTOR = factor
        logging.info(
            f"CLOSED_EDGES_FACTOR has been overwritten! Current value; {factor}"
        )

    @classmethod
    def enable_saving(cls, reason: str = ""):
        BaseConfig.DISABLE_SAVING = False
        BaseConfig.DISABLE_SAVING_NOTE = reason

    @classmethod
    def disable_saving(cls, reason: str = ""):
        BaseConfig.DISABLE_SAVING = True
        BaseConfig.DISABLE_SAVING_NOTE = reason

    @classmethod
    def update_synthetic_frame_size(cls, frame_size: Tuple[int, int]):
        cls.SYNTHETIC_FRAME_SIZE = frame_size
        logging.info(
            f"SYNTHETIC_FRAME_SIZE has been overwritten! Current value: {frame_size}"
        )

    @classmethod
    def _setup_logger(cls, log_level=None, details=None):
        log_level = log_level or logging.INFO
        details = details or ""
        _logger = logging.getLogger()
        if not _logger.hasHandlers():
            _logger.setLevel(log_level)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            logs_dir = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(logs_dir, f"project_{details}.log")
            )
            file_handler.setLevel(log_level)
            # noinspection SpellCheckingInspection
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            _logger.addHandler(console_handler)
            _logger.addHandler(file_handler)

    def __str__(self):
        def is_method_like(attr_value):
            import inspect

            return (
                inspect.isfunction(attr_value)
                or inspect.ismethod(attr_value)
                or isinstance(attr_value, staticmethod)
                or isinstance(attr_value, classmethod)
            )

        _class_vars = {
            k: v
            for k, v in vars(self.__class__).items()
            if not k.startswith("__") and not is_method_like(v)
        }
        _instance_vars = {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("__") and not callable(v)
        }
        _config_dict = {**_class_vars, **_instance_vars}
        return f"{self.__class__.__name__}: {_config_dict}"
