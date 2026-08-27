import logging
import os
import uuid
from typing import List, Optional, Tuple

from .dataset_id import DatasetId
from .file_definitions import (
    INPLACE_DIR,
    ORIGINAL_DIR,
    SYNTHETIC_DIR,
    RenderStyle,
    SaveSpec,
    save_csv,
    save_network_csv,
    save_pickle,
    save_text,
    save_webp,
)

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

    # IMAGE_SIZE: the input image's true size (height, width) in pixels, as it
    # is on disk.  Recorded, never laid out by: what the network and its
    # background are drawn in is FRAME_SIZE, which the image is scaled to.
    IMAGE_SIZE: Tuple[int, int] = None

    # FRAME_SIZE: Original network plotting frame (X_range, Y_range)
    FRAME_SIZE: Tuple[int, int] = None

    # SYNTHETIC_FRAME_SIZE: Synthetic network generation frame (X_range, Y_range)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = None

    CLOSED_NODES_FACTOR: float
    CLOSED_EDGES_FACTOR: float

    #: How far apart the values a sweep tries are, on both factors.
    SWEEP_STEP: float = 0.1

    #: Whether a sweep reports to wandb.  Off, it walks the grid itself and
    #: contacts nothing; the local report is written either way.
    USE_WANDB: bool = True

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

    # Hybrid tiling geometry: no values, so a config that omits one raises
    # rather than picking up a number buried in the pipeline body.
    TILE_FRAME_SIZE: Optional[Tuple[int, int]]
    TILE_FRAME_FACTOR: float
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
    SELECT_BEST: int = 0  # 0 = disabled; N = keep N best networks by metric distance

    # How each output is drawn.  Override with replace(BaseConfig.RENDER_X, ...).
    # The OpenCV renderers leave node_size/line_width unset: one pixel each.
    RENDER_ORIGINAL_GRAPH = RenderStyle(node_size=6.0, line_width=3.0, dpi=300)
    RENDER_SYNTHETIC_GRAPH = RenderStyle(
        node_size=6.0, line_width=3.0, dpi=300, show_on_the_fly=False
    )
    RENDER_BFS_SNAPSHOT = RenderStyle(node_size=6.0, line_width=3.0, dpi=300)
    RENDER_HYBRID_SNAPSHOT = RenderStyle()
    RENDER_HYBRID_GRAPH = RenderStyle()
    RENDER_MOSAIC_GRAPH = RenderStyle(border=True)
    RENDER_SCALED_GRAPH = RenderStyle()

    # What each output is written as.  Several specs = several files.
    SAVE_ORIGINAL_IMAGE = (SaveSpec(ORIGINAL_DIR, "original_image", "webp", save_webp),)
    SAVE_ORIGINAL_GRAPH = (SaveSpec(ORIGINAL_DIR, "original_graph", "webp", save_webp),)
    SAVE_ORIGINAL_NETWORK = (
        SaveSpec(ORIGINAL_DIR, "original_network", "csv", save_network_csv),
    )
    SAVE_ORIGINAL_PROPERTY = (
        SaveSpec(ORIGINAL_DIR, "original_property", "pkl", save_pickle),
    )
    SAVE_ORIGINAL_REPORT = (
        SaveSpec(ORIGINAL_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
    )
    SAVE_SYNTHETIC_NETWORK = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "pkl", save_pickle),
    )
    SAVE_SYNTHETIC_EXPORT = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "csv", save_network_csv),
    )
    SAVE_SYNTHETIC_REPORT = (
        SaveSpec(SYNTHETIC_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    # Written by every sweep, whether or not it talked to wandb: the trial
    # grid is the result, and it should not live only on someone's server.
    SAVE_SWEEP_REPORT = (
        SaveSpec(INPLACE_DIR, "sweep_report", "csv", save_csv, use_timestamp=False),
    )
    SAVE_ANALYSIS_DATA = (
        SaveSpec(INPLACE_DIR, "analysis_data", "pkl", save_pickle, use_timestamp=False),
    )
    SAVE_ANALYSIS_FIGURE = (
        SaveSpec(INPLACE_DIR, "analysis_figure", "webp", save_webp),
    )

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
    def initialize(cls) -> None:
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
    def render(cls, identifier: str) -> RenderStyle:
        """Return the :class:`RenderStyle` for *identifier*.

        Reads ``RENDER_<IDENTIFIER>``, so a mode's override resolves through the
        normal MRO.  Call this on the active config, not on ``BaseConfig``.
        """
        style = getattr(cls, f"RENDER_{identifier.upper()}", None)
        assert style is not None, (
            f"no render style for {identifier!r}; declare "
            f"RENDER_{identifier.upper()} on the config or on BaseConfig"
        )
        return style

    @classmethod
    def save(cls, identifier: str) -> Tuple[SaveSpec, ...]:
        """Return the :class:`SaveSpec` tuple for *identifier*.

        Reads ``SAVE_<IDENTIFIER>``, so a mode's override resolves through the
        normal MRO.  The twin of :meth:`render`; call it on the active config,
        not on ``BaseConfig``.
        """
        specs = getattr(cls, f"SAVE_{identifier.upper()}", None)
        if specs is None:
            raise ValueError(
                f"No save spec for '{identifier}' in {cls.__name__}. Declare "
                f"SAVE_{identifier.upper()} on the config or on BaseConfig."
            )
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
