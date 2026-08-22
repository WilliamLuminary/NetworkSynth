import json
import logging
import os
import uuid
from typing import List, Optional, Tuple

from .enums import DatasetId

logger = logging.getLogger(__name__)


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Recognised ``extra`` keys (``tag``, ``dataset``, ``percent``) are promoted
    to top-level fields so they can be filtered with ``jq``.

    ``percent`` is a machine-readable completion figure, present on progress
    records.  It exists so a caller tailing this file can drive a progress bar
    without parsing percentages out of the message text — see
    ``INTEGRATION_PLAN.md``.  Emit it with
    ``logger.info(msg, extra=tagged("PROGRESS", percent=pct))``.

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
        for key in ("tag", "dataset", "percent"):
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

    MAX_ATTEMPTS = 10
    ERROR_CHECKER: str = (
        "multifractal"  # quality-gate algorithm; "none" disables checking
    )
    ERROR_TOLERANCE = 0.15  # Generally should be 0.15
    MIN_TILE_NODES = 100

    # Master seed for reproducible generation.  None = unseeded, i.e. a
    # different network on every run (the historical behaviour).  When set,
    # each worker is given a distinct derived seed (SEED + worker index) so
    # candidates still differ from one another but the whole run repeats
    # identically.
    SEED: Optional[int] = None

    SNAPSHOT_INTERVAL: int = 0  # 0 = disabled; N = snapshot every N new nodes
    SNAPSHOT_PLOT_WORKERS: int = 20
    SELECT_BEST: int = 0  # 0 = disabled; N = keep N best networks by metric distance

    LOG_MEMORY: bool = False
    # Declared per config, never toggled at runtime: a mutator here would set
    # the flag on BaseConfig for every config in the process, and for every
    # forked child with it.
    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "output")
    ORIGINAL_NETWORK_FUNC = ORIGINAL_IMAGE_FUNC = load_idle
    NETWORKS_FUNC = load_idle

    MAX_WORKERS: int = 50

    @classmethod
    def get_max_workers(cls, num_tasks: int = None) -> int:
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(1, cpu_count // 2), cls.MAX_WORKERS)
        if num_tasks is not None:
            max_workers = min(max_workers, num_tasks)
        return max_workers

    @classmethod
    def get_snapshot_plot_workers(cls) -> int:
        cpu_count = os.cpu_count() or 1
        return min(max(1, cpu_count // 2), cls.SNAPSHOT_PLOT_WORKERS)

    @classmethod
    def initialize(cls):
        """Establish this run's identity.  Logging is set up by the entry point.

        ``RUN_ID`` is generated here rather than as a side effect of logging
        setup, because it names the run's output directory (see
        ``handlers.run_paths``) and is needed whether or not anything logs.
        """
        from handlers.run_logging import configure_console

        if not cls.RUN_ID:
            cls.RUN_ID = uuid.uuid4().hex[:8]
        configure_console()

        module_parts = cls.__module__.split(".")
        mode = next((p for p in module_parts if p.endswith("_mode")), None)
        cls.OUTPUT_DENOTE = f"{mode}_{cls.__name__}" if mode else cls.__name__

    @classmethod
    def save(cls, identifier: str):
        """Return the save specs for *identifier*.

        Checks for a ``save_<identifier>`` classmethod on *cls* first, which
        resolves a mode's override through the normal MRO, then falls back to
        ``DEFAULT_SAVE_SPECS`` in ``file_definitions``.  Call this on the active
        config, not on ``BaseConfig``.

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

    RUN_ID: str = ""

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
