# SPDX-License-Identifier: GPL-3.0-or-later
"""The shared loaders, through a config that adds no loader code of its own."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from networksynth.configs import DatasetId, GenerateConfig

pytestmark = pytest.mark.unit


def _npy_pair(directory, name, matrix="_mat.npy", positions="_pos.npy"):
    # Three nodes, one edge: the third is isolated and has to go on load.
    adjacency = csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(3, 3))
    np.save(directory / f"{name}{matrix}", np.array(adjacency, dtype=object))
    np.save(
        directory / f"{name}{positions}", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )


def _config(directory, **overrides):
    return GenerateConfig(
        DATASETS=[DatasetId("t")],
        BASE_INPUT_PATH=str(directory),
        FRAME_SIZE=(6, 4),
        SYNTHETIC_FRAME_SIZE=(6, 4),
        CLOSED_NODES_FACTOR=1.0,
        CLOSED_EDGES_FACTOR=1.0,
        MEASURE_WEIGHTED=False,
        **overrides,
    )


def test_a_config_with_no_loader_code_reads_the_old_npy_names(tmp_path):
    _npy_pair(tmp_path, "t")

    graph = _config(tmp_path).load_original_network(DatasetId("t"))

    # The isolated node is dropped, and (x, y) arrived as (row, col).
    assert graph.number_of_nodes() == 2
    assert sorted(map(tuple, graph.positions())) == [(2.0, 1.0), (4.0, 3.0)]


def test_the_current_npy_names_read_the_same_way(tmp_path):
    _npy_pair(tmp_path, "t", matrix="_adjacency.npy", positions="_positions.npy")

    assert (
        _config(tmp_path).load_original_network(DatasetId("t")).number_of_nodes() == 2
    )


def test_the_image_suffix_is_the_configs_to_choose(tmp_path):
    import cv2

    cv2.imwrite(str(tmp_path / "t.png"), np.zeros((4, 6), dtype=np.uint8))

    assert _config(tmp_path, IMAGE_SUFFIX=".png").load_original_image(
        DatasetId("t")
    ).shape == (4, 6)
    assert _config(tmp_path).load_original_image(DatasetId("t")) is None
