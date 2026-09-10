# SPDX-License-Identifier: GPL-3.0-or-later
"""One frozen dataclass per pipeline; a config file instantiates one of them.

A field without a default is required: leaving it out fails when the config is
built, and so does naming a field the pipeline never reads.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

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

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
)
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "input")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output")


@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    MODE: ClassVar[str] = ""

    DATASETS: List[DatasetId]
    BASE_INPUT_PATH: str = DEFAULT_INPUT_PATH
    BASE_OUTPUT_PATH: str = DEFAULT_OUTPUT_PATH
    OUTPUT_DENOTE: str = ""
    RUN_ID: str = ""

    DISABLE_SAVING: bool = False
    DISABLE_SAVING_NOTE: str = ""
    LOG_MEMORY: bool = False
    MAX_WORKERS: int = 50

    SAVE_ORIGINAL_IMAGE: Tuple[SaveSpec, ...] = (
        SaveSpec(ORIGINAL_DIR, "original_image", "webp", save_webp),
    )
    SAVE_ORIGINAL_GRAPH: Tuple[SaveSpec, ...] = (
        SaveSpec(ORIGINAL_DIR, "original_graph", "webp", save_webp),
    )
    SAVE_ORIGINAL_NETWORK: Tuple[SaveSpec, ...] = (
        SaveSpec(ORIGINAL_DIR, "original_network", "csv", save_network_csv),
    )
    SAVE_ORIGINAL_PROPERTY: Tuple[SaveSpec, ...] = (
        SaveSpec(ORIGINAL_DIR, "original_property", "json", save_json),
    )
    SAVE_ORIGINAL_REPORT: Tuple[SaveSpec, ...] = (
        SaveSpec(ORIGINAL_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    SAVE_SYNTHETIC_GRAPH: Tuple[SaveSpec, ...] = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
    )
    SAVE_SYNTHETIC_NETWORK: Tuple[SaveSpec, ...] = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_network", "csv", save_network_csv),
    )
    SAVE_SYNTHETIC_REPORT: Tuple[SaveSpec, ...] = (
        SaveSpec(SYNTHETIC_DIR, "report", "txt", save_text, use_timestamp=False),
    )
    SAVE_SWEEP_REPORT: Tuple[SaveSpec, ...] = (
        SaveSpec(INPLACE_DIR, "sweep_report", "csv", save_csv, use_timestamp=False),
    )
    SAVE_ANALYSIS_DATA: Tuple[SaveSpec, ...] = (
        SaveSpec(INPLACE_DIR, "analysis_data", "json", save_json, use_timestamp=False),
    )
    SAVE_ANALYSIS_FIGURE: Tuple[SaveSpec, ...] = (
        SaveSpec(INPLACE_DIR, "analysis_figure", "webp", save_webp),
    )

    def __post_init__(self) -> None:
        if not self.MODE:
            raise ValueError(
                f"{type(self).__name__} claims no pipeline; build one of "
                "GenerateConfig, HybridConfig, SweepConfig or CompareConfig"
            )
        if not self.RUN_ID:
            object.__setattr__(self, "RUN_ID", uuid.uuid4().hex[:8])
        if not self.OUTPUT_DENOTE:
            object.__setattr__(self, "OUTPUT_DENOTE", self.MODE)

    def get_max_workers(self, num_tasks: int = None) -> int:
        from networksynth.utils import worker_count

        return worker_count(num_tasks, ceiling=self.MAX_WORKERS)

    def render(self, identifier: str) -> RenderStyle:
        return getattr(self, f"RENDER_{identifier.upper()}")

    def save(self, identifier: str) -> Tuple[SaveSpec, ...]:
        return getattr(self, f"SAVE_{identifier.upper()}")


@dataclass(frozen=True, kw_only=True)
class SynthesisConfig(BaseConfig):
    """What every pipeline that grows networks from an original reads."""

    FRAME_SIZE: Tuple[int, int]
    SYNTHETIC_FRAME_SIZE: Tuple[int, int]
    MEASURE_WEIGHTED: bool
    FULL_Q_BAND: bool = False

    SYNTHETIC_NETWORK_NUMBER: int = 0
    MAX_ATTEMPTS: int = 10
    ERROR_CHECKER: str = "multifractal"
    ERROR_TOLERANCE: float = 0.15
    SEED: Optional[int] = None

    IMAGE_SUFFIX: str = _IMAGE_SUFFIX
    RENDER_ORIGINAL_GRAPH: RenderStyle = RenderStyle(
        node_size=6.0, line_width=3.0, dpi=300
    )

    def load_original_network(self, dataset_id: DatasetId):
        from .loaders import load_network

        graph, _ = load_network(self.BASE_INPUT_PATH, dataset_id.path)
        return graph

    def load_original_image(self, dataset_id: DatasetId):
        from .loaders import load_image

        return load_image(
            self.BASE_INPUT_PATH, dataset_id.path, self.FRAME_SIZE, self.IMAGE_SUFFIX
        )


@dataclass(frozen=True, kw_only=True)
class GenerateConfig(SynthesisConfig):
    MODE: ClassVar[str] = "generate"

    CLOSED_NODES_FACTOR: float
    CLOSED_EDGES_FACTOR: float

    SYNTHETIC_GRAPH_NUMBER: int = 0
    SNAPSHOT_INTERVAL: int = 0
    SELECT_BEST: int = 0
    SNAPSHOT_PLOT_WORKERS: int = 20

    RENDER_SYNTHETIC_GRAPH: RenderStyle = RenderStyle(
        node_size=6.0, line_width=3.0, dpi=300, show_on_the_fly=False
    )
    RENDER_BFS_SNAPSHOT: RenderStyle = RenderStyle(
        node_size=6.0, line_width=3.0, dpi=300
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.DISABLE_SAVING and self.SNAPSHOT_INTERVAL:
            raise ValueError(
                f"SNAPSHOT_INTERVAL={self.SNAPSHOT_INTERVAL} needs saving enabled: "
                "snapshots bypass the Saver, so a run with DISABLE_SAVING set "
                "would still write them.  Set one or the other."
            )

    def snapshot_formats(self) -> tuple:
        return tuple(spec.extension for spec in self.SAVE_SYNTHETIC_GRAPH)

    def get_snapshot_plot_workers(self) -> int:
        from networksynth.utils import worker_count

        return worker_count(ceiling=self.SNAPSHOT_PLOT_WORKERS)


@dataclass(frozen=True, kw_only=True)
class HybridConfig(GenerateConfig):
    MODE: ClassVar[str] = "hybrid"

    TARGET_SCALE: Tuple[int, int]
    NUM_CENTERS: int
    MIN_CENTER_DISTANCE_FACTOR: float
    TILE_FRAME_FACTOR: float
    TILE_FRAME_SIZE: Optional[Tuple[int, int]] = None
    DATASET_FACTORS: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    PHASE2_MAX_ROUNDS: int = 500
    MIN_TILE_NODES: int = 100

    RENDER_HYBRID_SNAPSHOT: RenderStyle = RenderStyle()
    RENDER_HYBRID_GRAPH: RenderStyle = RenderStyle()


@dataclass(frozen=True, kw_only=True)
class SweepConfig(SynthesisConfig):
    MODE: ClassVar[str] = "sweep"

    NF_RANGE: Tuple[float, float]
    EF_RANGE: Tuple[float, float]
    SWEEP_STEP: float = 0.1
    USE_WANDB: bool = True


@dataclass(frozen=True, kw_only=True)
class CompareConfig(BaseConfig):
    MODE: ClassVar[str] = "compare"

    ORIGINAL_NETWORKS_PATH: str
    SYNTHETIC_NETWORKS_PATH: str
    MEASURE_WEIGHTED: bool
    FULL_Q_BAND: bool = False

    def load_networks(self, path: str) -> list:
        from networksynth.graphs import load_graphs

        return load_graphs(path)


CONFIG_CLASSES = {
    cls.MODE: cls for cls in (GenerateConfig, HybridConfig, SweepConfig, CompareConfig)
}
