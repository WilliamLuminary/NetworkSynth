# SPDX-License-Identifier: GPL-3.0-or-later
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
    save_json,
    save_network_csv,
    save_text,
    save_webp,
)
from .loaders import IMAGE_SUFFIX as _IMAGE_SUFFIX

logger = logging.getLogger(__name__)


def _unwired(cls, func) -> bool:
    """Whether initialize() should wire a loader onto this class.

    None means nothing was set. A method bound to an ancestor means that
    ancestor's initialize() ran first and this class inherited its binding,
    which would read the ancestor's paths; it needs its own. Anything else is a
    loader somebody plugged in, and stays.
    """
    if func is None:
        return True
    owner = getattr(func, "__self__", None)
    return isinstance(owner, type) and owner is not cls and issubclass(cls, owner)


class BaseConfig:

    MODE: str = ""

    DATASETS: Optional[List[DatasetId]] = None

    @classmethod
    def get_datasets(cls) -> List[DatasetId]:
        if cls.DATASETS is None:
            raise ValueError("DATASETS must be defined in the config")
        return cls.DATASETS

    IMAGE_SIZE: Tuple[int, int] = None

    FRAME_SIZE: Tuple[int, int] = None

    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = None

    CLOSED_NODES_FACTOR: float
    CLOSED_EDGES_FACTOR: float

    SWEEP_STEP: float = 0.1

    USE_WANDB: bool = True

    MEASURE_WEIGHTED: bool
    FULL_Q_BAND: bool = False

    SYNTHETIC_GRAPH_NUMBER: int = 0
    SYNTHETIC_NETWORK_NUMBER: int = 0

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    BASE_INPUT_PATH = os.path.join(BASE_DATA_PATH, "input")

    ORIGINAL_NETWORKS_PATH = None
    SYNTHETIC_NETWORKS_PATH = None

    MAX_ATTEMPTS = 10
    ERROR_CHECKER: str = "multifractal"
    ERROR_TOLERANCE = 0.15
    MIN_TILE_NODES = 100
    PHASE2_MAX_ROUNDS = 500

    # Annotated with no value: a config that omits one raises on access rather
    # than picking up a number buried in the pipeline body.
    TILE_FRAME_SIZE: Optional[Tuple[int, int]]
    TILE_FRAME_FACTOR: float
    MIN_CENTER_DISTANCE_FACTOR: float
    NUM_CENTERS: int
    DATASET_FACTORS: dict

    SEED: Optional[int] = None

    SNAPSHOT_INTERVAL: int = 0
    SNAPSHOT_PLOT_WORKERS: int = 20
    SELECT_BEST: int = 0

    RENDER_ORIGINAL_GRAPH = RenderStyle(node_size=6.0, line_width=3.0, dpi=300)
    RENDER_SYNTHETIC_GRAPH = RenderStyle(
        node_size=6.0, line_width=3.0, dpi=300, show_on_the_fly=False
    )
    RENDER_BFS_SNAPSHOT = RenderStyle(node_size=6.0, line_width=3.0, dpi=300)
    RENDER_HYBRID_SNAPSHOT = RenderStyle()
    RENDER_HYBRID_GRAPH = RenderStyle()

    SAVE_ORIGINAL_IMAGE = (SaveSpec(ORIGINAL_DIR, "original_image", "webp", save_webp),)
    SAVE_ORIGINAL_GRAPH = (SaveSpec(ORIGINAL_DIR, "original_graph", "webp", save_webp),)
    SAVE_ORIGINAL_NETWORK = (
        SaveSpec(ORIGINAL_DIR, "original_network", "csv", save_network_csv),
    )
    SAVE_ORIGINAL_PROPERTY = (
        SaveSpec(ORIGINAL_DIR, "original_property", "json", save_json),
    )
    SAVE_ORIGINAL_REPORT = (
        SaveSpec(ORIGINAL_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
    )
    SAVE_SYNTHETIC_NETWORK = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "csv", save_network_csv),
    )
    SAVE_SYNTHETIC_REPORT = (
        SaveSpec(SYNTHETIC_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    SAVE_SWEEP_REPORT = (
        SaveSpec(INPLACE_DIR, "sweep_report", "csv", save_csv, use_timestamp=False),
    )
    SAVE_ANALYSIS_DATA = (
        SaveSpec(INPLACE_DIR, "analysis_data", "json", save_json, use_timestamp=False),
    )
    SAVE_ANALYSIS_FIGURE = (
        SaveSpec(INPLACE_DIR, "analysis_figure", "webp", save_webp),
    )

    LOG_MEMORY: bool = False
    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""

    BASE_OUTPUT_PATH = os.path.join(BASE_DATA_PATH, "output")
    # A config plugs in a loader by overriding load_original_* below, or by
    # assigning any callable here; initialize() fills in only what is still None.
    ORIGINAL_NETWORK_FUNC = ORIGINAL_IMAGE_FUNC = None
    NETWORKS_FUNC = None
    IMAGE_SUFFIX: str = _IMAGE_SUFFIX

    MAX_WORKERS: int = 50

    @classmethod
    def get_max_workers(cls, num_tasks: int = None) -> int:
        from networksynth.utils import worker_count

        return worker_count(num_tasks, ceiling=cls.MAX_WORKERS)

    @classmethod
    def snapshot_formats(cls) -> tuple:
        return tuple(spec.extension for spec in cls.SAVE_SYNTHETIC_GRAPH)

    @classmethod
    def get_snapshot_plot_workers(cls) -> int:
        from networksynth.utils import worker_count

        return worker_count(ceiling=cls.SNAPSHOT_PLOT_WORKERS)

    @classmethod
    def initialize(cls) -> None:
        from networksynth.handlers.run_logging import configure_console

        if not cls.MODE:
            raise ValueError(
                f"{cls.__name__} sets no MODE, so no pipeline claims it. Set "
                "MODE to the pipeline that runs this config."
            )

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

        if _unwired(cls, cls.ORIGINAL_NETWORK_FUNC):
            cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        if _unwired(cls, cls.ORIGINAL_IMAGE_FUNC):
            cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        if _unwired(cls, cls.NETWORKS_FUNC):
            from networksynth.graphs import load_graphs

            cls.NETWORKS_FUNC = staticmethod(load_graphs)

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId):
        from .loaders import load_network

        graph, _ = load_network(cls.BASE_INPUT_PATH, dataset_id.path)
        return graph

    @classmethod
    def load_original_image(cls, dataset_id: DatasetId):
        from .loaders import load_image

        return load_image(
            cls.BASE_INPUT_PATH, dataset_id.path, cls.FRAME_SIZE, cls.IMAGE_SUFFIX
        )

    @classmethod
    def render(cls, identifier: str) -> RenderStyle:
        return getattr(cls, f"RENDER_{identifier.upper()}")

    @classmethod
    def save(cls, identifier: str) -> Tuple[SaveSpec, ...]:
        return getattr(cls, f"SAVE_{identifier.upper()}")

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
