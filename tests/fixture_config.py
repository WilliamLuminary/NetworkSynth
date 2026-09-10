# SPDX-License-Identifier: GPL-3.0-or-later
"""Configs written for the tests.

A real config's values belong to whoever tunes it, so a test that asserts on one
stops testing what it claims the moment somebody retunes it. Tests build on
these instead. The networks come from tests/data, the same fixtures conftest
serves, so no test needs the sample data under data/input either.

The exception is a test whose subject *is* the shipped configs, such as
TestEveryConfigInTheRepoLoads. That one must walk the real ones.
"""

import os
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from networksynth.configs.base_config import BaseConfig
from networksynth.configs.dataset_id import DatasetId
from networksynth.configs.file_definitions import (
    SYNTHETIC_DIR,
    SaveSpec,
    save_png,
    save_webp,
)
from networksynth.graphs.graphml_io import read_graph_graphml
from networksynth.graphs.synth_graph import SynthGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNWEIGHTED_NETWORK = os.path.join(DATA_DIR, "sample1_unweighted_network.graphml")
WEIGHTED_NETWORK = os.path.join(DATA_DIR, "A_10kX_weighted_network_positive.graphml")

FIXTURE_DATASET = DatasetId("fixture")


class FixtureConfig(BaseConfig):
    """Everything a pipeline reads, set small enough to run in a test.

    NETWORK_FILE is the seam: point a subclass at the weighted fixture when the
    behaviour under test needs weights.
    """

    MODE = "generate"
    DATASETS = [FIXTURE_DATASET]
    NETWORK_FILE = UNWEIGHTED_NETWORK

    IMAGE_SIZE: Tuple[int, int] = (512, 512)
    FRAME_SIZE: Tuple[int, int] = (512, 512)
    # 256 yields ~228 nodes from the unweighted fixture; generate_network
    # rejects anything under 100, and 128 fell short of it.
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (256, 256)

    # Not on BaseConfig, and SynthParams.from_config demands both.
    CLOSED_NODES_FACTOR = 1.0
    CLOSED_EDGES_FACTOR = 1.0

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    MAX_ATTEMPTS = 1
    ERROR_TOLERANCE = 0.15
    ERROR_CHECKER = "none"
    MEASURE_WEIGHTED = False
    FULL_Q_BAND = False

    SNAPSHOT_INTERVAL = 0
    LOG_MEMORY = False
    SEED = 1234

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @classmethod
    def load_original_network(cls, dataset_id: DatasetId) -> SynthGraph:
        return read_graph_graphml(cls.NETWORK_FILE)

    @staticmethod
    def load_original_image(dataset_id: DatasetId) -> Optional[np.ndarray]:
        return None


class FixtureWeightedConfig(FixtureConfig):
    NETWORK_FILE = WEIGHTED_NETWORK
    MEASURE_WEIGHTED = True


class FixtureHybridConfig(FixtureConfig):
    MODE = "hybrid"

    TARGET_SCALE: Tuple[int, int] = (1, 2)
    NUM_CENTERS: int = 2
    PHASE2_MAX_ROUNDS: int = 10
    MIN_CENTER_DISTANCE_FACTOR: float = 1.5
    TILE_FRAME_SIZE: Tuple[int, int] = (510, 510)
    TILE_FRAME_FACTOR: float = 0.5
    DATASET_FACTORS: Dict[str, Tuple[float, float]] = {}

    IMAGE_SIZE: Tuple[int, int] = (510, 510)
    FRAME_SIZE: Tuple[int, int] = (510, 510)
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)

    SYNTHETIC_GRAPH_NUMBER = 0
    SYNTHETIC_NETWORK_NUMBER = 0

    # Two formats, so a test can assert a mode config adds to the base's one.
    SAVE_SYNTHETIC_GRAPH = (
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "png", save_png),
    )


class FixtureSnapshotConfig(FixtureConfig):
    SYNTHETIC_FRAME_SIZE: Tuple[int, int] = (510, 510)
    SNAPSHOT_INTERVAL = 10
    SELECT_BEST = 0

    RENDER_BFS_SNAPSHOT = replace(BaseConfig.RENDER_BFS_SNAPSHOT, dpi=72)


class FixtureSweepConfig(FixtureConfig):
    MODE = "sweep"

    NF_RANGE: Tuple[float, float] = (0.3, 2.0)
    EF_RANGE: Tuple[float, float] = (0.3, 2.0)
    DISABLE_SAVING = True
    DISABLE_SAVING_NOTE = "Fixture sweep"


class FixtureCompareConfig(FixtureConfig):
    MODE = "compare"
    MEASURE_WEIGHTED = False

    ORIGINAL_NETWORKS_PATH = ""
    SYNTHETIC_NETWORKS_PATH = ""
