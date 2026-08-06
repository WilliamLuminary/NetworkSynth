# src/configs/base_config.py
import json
import logging
import os
import random
import uuid
from typing import List, Optional, Tuple

import numpy as np

from .enums import DatasetId

logger = logging.getLogger(__name__)


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Recognised ``extra`` keys (``tag``, ``dataset``) are promoted to
    top-level fields so they can be filtered with ``jq``.

    Each entry includes a ``run_id`` so concurrent runs appending to the
    same file can be distinguished: ``jq 'select(.run_id == "abc123")'``.
    """

    def __init__(self, run_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.default_time_format),
            "run_id": self.run_id,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("tag", "dataset"):
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def tagged(tag: str, **extra) -> dict:
    """Return an ``extra`` dict for use with ``logger.info(msg, extra=tagged("PHASE1"))``.

    Usage::

        logger.info("Phase 1 complete", extra=tagged("PHASE1", dataset="sample_A"))

    The tag and any additional keys are embedded in the JSON log output
    and ignored by the plain-text console formatter.
    """
    return {"tag": tag, **extra}


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
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, "input")

    NETWORKS_DATA_PATH = None
    ATTRIBUTES_DICT_DATA_PATH = None

    MAX_ATTEMPTS = 10
    RANDOM_SEED: Optional[int] = 42  # None = seed from OS entropy (non-reproducible)
    ERROR_CHECKER: str = (
        "multifractal"  # quality-gate algorithm; "none" disables checking
    )
    ERROR_TOLERANCE = 0.15  # Generally should be 0.15
    MIN_TILE_NODES = 100

    SNAPSHOT_INTERVAL: int = 0  # 0 = disabled; N = snapshot every N new nodes
    SNAPSHOT_PLOT_WORKERS: int = 20
    SELECT_BEST: int = 0  # 0 = disabled; N = keep N best networks by metric distance

    LOG_MEMORY: bool = False
    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "output")
    ORIGINAL_NETWORK_FUNC = ORIGINAL_IMAGE_FUNC = load_idle
    NETWORKS_FUNC = load_idle
    ATTRIBUTES_DICT_FUNC = load_idle

    MAX_WORKERS: int = 50

    @classmethod
    def get_max_workers(cls, num_tasks: int = None) -> int:
        """Return number of worker processes, capped by half the CPUs and MAX_WORKERS."""
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(1, cpu_count // 2), cls.MAX_WORKERS)
        if num_tasks is not None:
            max_workers = min(max_workers, num_tasks)
        return max_workers

    @classmethod
    def seed_rng(cls, index: int) -> None:
        """Seed this process's RNGs deterministically from ``RANDOM_SEED``.

        Called at the top of every worker task. ``index`` must be unique
        within a batch, otherwise sibling workers draw the same numbers and
        produce identical networks. When ``RANDOM_SEED`` is None the RNGs are
        left on their OS-entropy seeding, so runs are not reproducible.
        """
        if cls.RANDOM_SEED is None:
            return
        seed = (cls.RANDOM_SEED + index) % (2**31)
        random.seed(seed)
        np.random.seed(seed)

    @classmethod
    def get_snapshot_plot_workers(cls) -> int:
        """Return number of snapshot plotting workers, capped by half the CPUs and SNAPSHOT_PLOT_WORKERS."""
        cpu_count = os.cpu_count() or 1
        return min(max(1, cpu_count // 2), cls.SNAPSHOT_PLOT_WORKERS)

    @classmethod
    def initialize(cls):
        cls._setup_logger(details=cls.__name__)
        module_parts = cls.__module__.split(".")
        mode = next((p for p in module_parts if p.endswith("_mode")), None)
        cls.OUTPUT_DENOTE = f"{mode}_{cls.__name__}" if mode else cls.__name__

    @classmethod
    def save(cls, identifier: str):
        """Return the save specs for *identifier*.

        Checks for a ``save_<identifier>`` classmethod first (mode
        overrides injected by ``_inject_dependencies``), then falls
        back to ``DEFAULT_SAVE_SPECS`` in ``file_definitions``.

        Each spec is a tuple::

            (relative_dir, detail, extension, save_fn[, use_timestamp])
        """
        method = getattr(cls, f"save_{identifier}", None)
        if method is not None:
            return method()
        from .file_definitions import DEFAULT_SAVE_SPECS

        specs = DEFAULT_SAVE_SPECS.get(identifier)
        if specs is None:
            raise ValueError(f"No save spec for '{identifier}' in {cls.__name__}.")
        return specs

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
            f"CLOSED_NODES_FACTOR has been overwritten! Current value: {factor}",
            extra=tagged("CONFIG"),
        )

    @classmethod
    def set_edge_factor(cls, factor: float):
        cls.CLOSED_EDGES_FACTOR = factor
        logging.info(
            f"CLOSED_EDGES_FACTOR has been overwritten! Current value; {factor}",
            extra=tagged("CONFIG"),
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
            f"SYNTHETIC_FRAME_SIZE has been overwritten! Current value: {frame_size}",
            extra=tagged("CONFIG"),
        )

    _logger_initialized = False
    RUN_ID: str = ""

    @classmethod
    def _setup_logger(cls, log_level=None, details=None):
        if BaseConfig._logger_initialized:
            return
        log_level = log_level or logging.INFO
        details = details or ""

        BaseConfig.RUN_ID = uuid.uuid4().hex[:8]

        _logger = logging.getLogger()
        _logger.setLevel(log_level)

        # Console: human-readable plain text
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        # noinspection SpellCheckingInspection
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File: JSON lines (one JSON object per line, queryable with jq)
        logs_dir = os.path.join(BaseConfig.BASE_OUTPUT_PATH, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f"project_{details}.jsonl"),
            mode="a",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(_JsonFormatter(run_id=BaseConfig.RUN_ID))

        _logger.addHandler(console_handler)
        _logger.addHandler(file_handler)
        BaseConfig._logger_initialized = True

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
