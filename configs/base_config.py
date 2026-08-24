import logging
import os
import uuid
from typing import List, Optional, Tuple

from .dataset_id import DatasetId

logger = logging.getLogger(__name__)


def load_idle(*_, **__):
    err_msg = "Logic of loading data is not implemented yet."
    logging.info(err_msg)
    raise NotImplementedError(err_msg)


class BaseConfig:
    """Define base paths, this config file must be in the subdirectory of the project root"""

    #: Which pipeline runs this config.  One of ``pipelines.PIPELINES``; a
    #: config that does not say cannot be run, because nothing else decides.
    MODE: str = ""

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
    FULL_Q_BAND: bool = False

    SYNTHETIC_GRAPH_NUMBER: int = 0
    SYNTHETIC_NETWORK_NUMBER: int = 0

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, "input")

    ORIGINAL_NETWORKS_PATH = None
    SYNTHETIC_NETWORKS_PATH = None

    MAX_ATTEMPTS = 10
    ERROR_CHECKER: str = (
        "multifractal"  # quality-gate algorithm; "none" disables checking
    )
    ERROR_TOLERANCE = 0.15  # Generally should be 0.15
    MIN_TILE_NODES = 100

    # Hybrid tiling geometry, declared without values so a config that omits
    # one raises instead of picking up a number buried in the pipeline body.
    # TILE_FRAME_SIZE None = auto: nearest-neighbour distance *
    # TILE_FRAME_FACTOR, floored at MIN_TILE_FRAME.  NUM_CENTERS 0 = no cap.
    TILE_FRAME_SIZE: Optional[Tuple[int, int]]
    TILE_FRAME_FACTOR: float
    MIN_TILE_FRAME: float
    MIN_CENTER_DISTANCE_FACTOR: float
    NUM_CENTERS: int
    DATASET_FACTORS: dict

    # Master seed for reproducible generation.  None = unseeded, i.e. a
    # different network on every run (the historical behaviour).  When set,
    # each worker is given a distinct derived seed (SEED + worker index) so
    # candidates still differ from one another but the whole run repeats
    # identically.
    SEED: Optional[int] = None

    # Snapshots of generation in progress.  The only switch: a mode does not
    # get its own pipeline for them, it sets these.
    #   0      disabled
    #   N > 0  every N (new nodes in generate, Phase 2 rounds in hybrid)
    #   N < 0  hybrid only — roughly |N| log-spaced snapshots across the run
    SNAPSHOT_INTERVAL: int = 0
    SNAPSHOT_PLOT_WORKERS: int = 20
    #: Rendering style for hybrid Phase 2 snapshots: dpi, node_size, line_width.
    #: Empty means the renderer's own defaults.
    HYBRID_SNAPSHOT_STYLE: dict = {}
    #: Rendering style for generate snapshots, same keys as above.
    PLOT_STYLE: dict = {}
    #: Cap on a rendered image's longest side in pixels.  None = no cap.
    RENDER_MAX_PX: Optional[int] = None
    ORIGINAL_GRAPH_NODE_SCALE: float = 1.0
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

        if not cls.MODE:
            raise ValueError(
                f"{cls.__name__} sets no MODE, so no pipeline claims it. Set "
                "MODE to the pipeline that runs this config."
            )

        # Snapshots are written straight to disk by the render pool, bypassing
        # the Saver, so DISABLE_SAVING cannot suppress them.  Rather than let a
        # "disabled" run litter the output directory, refuse the combination.
        if cls.DISABLE_SAVING and cls.SNAPSHOT_INTERVAL:
            raise ValueError(
                f"SNAPSHOT_INTERVAL={cls.SNAPSHOT_INTERVAL} needs saving enabled: "
                "snapshots bypass the Saver, so a run with DISABLE_SAVING set "
                "would still write them.  Set one or the other."
            )

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
