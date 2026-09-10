# SPDX-License-Identifier: GPL-3.0-or-later
"""Configs written for the tests.

A real config's values belong to whoever tunes it, so a test that asserts on one
stops testing what it claims the moment somebody retunes it. Tests build on
these instead, with dataclasses.replace for what they need changed. The networks
come from tests/data, the same fixtures conftest serves, so no test needs the
sample data under data/input either.

The exception is a test whose subject *is* the shipped configs, such as
TestEveryConfigInTheRepoLoads. That one must walk the real ones.
"""

import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from networksynth.configs import (
    CompareConfig,
    GenerateConfig,
    HybridConfig,
    SweepConfig,
)
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


@dataclass(frozen=True, kw_only=True)
class _GraphmlInput:
    """Reads one GraphML file whatever the dataset; NETWORK_FILE is the seam."""

    NETWORK_FILE: str = UNWEIGHTED_NETWORK

    def load_original_network(self, dataset_id: DatasetId) -> SynthGraph:
        return read_graph_graphml(self.NETWORK_FILE)

    def load_original_image(self, dataset_id: DatasetId) -> Optional[np.ndarray]:
        return None


@dataclass(frozen=True, kw_only=True)
class FixtureGenerateConfig(_GraphmlInput, GenerateConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class FixtureHybridConfig(_GraphmlInput, HybridConfig):
    pass


@dataclass(frozen=True, kw_only=True)
class FixtureSweepConfig(_GraphmlInput, SweepConfig):
    pass


# 256 yields ~228 nodes from the unweighted fixture; generate_network rejects
# anything under 100, and 128 fell short of it.
FIXTURE = FixtureGenerateConfig(
    DATASETS=[FIXTURE_DATASET],
    FRAME_SIZE=(512, 512),
    SYNTHETIC_FRAME_SIZE=(256, 256),
    CLOSED_NODES_FACTOR=1.0,
    CLOSED_EDGES_FACTOR=1.0,
    SYNTHETIC_GRAPH_NUMBER=1,
    SYNTHETIC_NETWORK_NUMBER=1,
    MAX_ATTEMPTS=1,
    ERROR_CHECKER="none",
    MEASURE_WEIGHTED=False,
    SEED=1234,
)

FIXTURE_WEIGHTED = replace(
    FIXTURE, NETWORK_FILE=WEIGHTED_NETWORK, MEASURE_WEIGHTED=True
)

FIXTURE_HYBRID = FixtureHybridConfig(
    DATASETS=[FIXTURE_DATASET],
    FRAME_SIZE=(510, 510),
    SYNTHETIC_FRAME_SIZE=(510, 510),
    CLOSED_NODES_FACTOR=1.0,
    CLOSED_EDGES_FACTOR=1.0,
    TARGET_SCALE=(1, 2),
    NUM_CENTERS=2,
    PHASE2_MAX_ROUNDS=10,
    MIN_CENTER_DISTANCE_FACTOR=1.5,
    TILE_FRAME_SIZE=(510, 510),
    TILE_FRAME_FACTOR=0.5,
    MAX_ATTEMPTS=1,
    ERROR_CHECKER="none",
    MEASURE_WEIGHTED=False,
    SEED=1234,
    # Two formats, so a test can assert a config adds to the default one.
    SAVE_SYNTHETIC_GRAPH=(
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "webp", save_webp),
        SaveSpec(SYNTHETIC_DIR, "synthetic_graph", "png", save_png),
    ),
)

FIXTURE_SNAPSHOT = replace(
    FIXTURE,
    SYNTHETIC_FRAME_SIZE=(510, 510),
    SNAPSHOT_INTERVAL=10,
    RENDER_BFS_SNAPSHOT=replace(FIXTURE.RENDER_BFS_SNAPSHOT, dpi=72),
)

FIXTURE_SWEEP = FixtureSweepConfig(
    DATASETS=[FIXTURE_DATASET],
    FRAME_SIZE=(512, 512),
    SYNTHETIC_FRAME_SIZE=(256, 256),
    NF_RANGE=(0.3, 2.0),
    EF_RANGE=(0.3, 2.0),
    SYNTHETIC_NETWORK_NUMBER=1,
    MAX_ATTEMPTS=1,
    ERROR_CHECKER="none",
    MEASURE_WEIGHTED=False,
    SEED=1234,
    DISABLE_SAVING=True,
    DISABLE_SAVING_NOTE="Fixture sweep",
    USE_WANDB=False,
)

FIXTURE_COMPARE = CompareConfig(
    DATASETS=[FIXTURE_DATASET],
    MEASURE_WEIGHTED=False,
    ORIGINAL_NETWORKS_PATH="",
    SYNTHETIC_NETWORKS_PATH="",
)

FIXTURES = {
    "generate": FIXTURE,
    "hybrid": FIXTURE_HYBRID,
    "sweep": FIXTURE_SWEEP,
    "compare": FIXTURE_COMPARE,
}
