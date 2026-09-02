# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import os
import re
from typing import List, Optional

import cv2
import numpy as np

from networksynth.graphs.synth_graph import SynthGraph
from networksynth.utils import (
    find_file_with_pattern,
    resize_image,
    transpose_positions,
    trim_image,
)

from ..base_config import BaseConfig
from ..dataset_id import DatasetId

logger = logging.getLogger(__name__)


def _generate_new_datasets() -> List[DatasetId]:
    set_names = ["004", "008", "011", "014", "017", "020", "023", "026", "029", "032"]
    return [DatasetId(s) for s in set_names]


class Config2(BaseConfig):
    MODE = "generate"

    DATASETS = [DatasetId("004")]

    IMAGE_SIZE = (1887, 2048)
    FRAME_SIZE = (1887 // 4, 2048 // 4)
    SYNTHETIC_FRAME_SIZE = FRAME_SIZE

    CLOSED_NODES_FACTOR = 1.2
    CLOSED_EDGES_FACTOR = 0.8

    SYNTHETIC_GRAPH_NUMBER = 1
    SYNTHETIC_NETWORK_NUMBER = 1

    MEASURE_WEIGHTED = False
    ERROR_TOLERANCE = 0.3

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, "new_input")
    POSITION_DATA_DIR = os.path.join(BASE_INPUT_PATH, "position")
    ADJ_MATRIX_DATA_DIR = os.path.join(BASE_INPUT_PATH, "sparse_matrices")
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, "Original Graphs")

    @classmethod
    def initialize(cls) -> None:
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image

    @staticmethod
    def load_original_network(dataset_id: DatasetId) -> SynthGraph:
        positions = _load_positions(dataset_id)
        mat = _load_sparse_matrix(dataset_id)
        from networksynth.utils import build_graph

        original_network = build_graph(positions, mat)
        transpose_positions(original_network)
        return original_network

    @staticmethod
    def load_original_image(dataset_id: DatasetId) -> Optional[np.ndarray]:
        image = _load_raw_image(dataset_id)
        image = trim_image(image)
        image = resize_image(image, Config2.FRAME_SIZE)
        return image


def _load_positions(dataset_id: DatasetId) -> np.ndarray:
    set_name = dataset_id[0]
    directory_path = Config2.POSITION_DATA_DIR
    file_path = find_file_with_pattern(
        directory_path,
        rf"W-\d+-\d+-\d+_{re.escape(set_name)}_postion\.npy",
        f"positions_of_nodes for {set_name}",
    )
    logger.info(f"Positions file loaded: {file_path}")
    positions = np.load(file_path, allow_pickle=True)
    return positions


def _load_sparse_matrix(dataset_id: DatasetId):
    set_name = dataset_id[0]
    directory_path = Config2.ADJ_MATRIX_DATA_DIR
    file_path = find_file_with_pattern(
        directory_path, r"sparse_matrices\.npz", details="sparse matrix"
    )
    matrix_data = np.load(file_path, allow_pickle=True)

    key_pattern = rf"W-\d+-\d+-\d+_{re.escape(set_name)}_EL"
    matching_keys = [
        key for key in matrix_data.keys() if re.fullmatch(key_pattern, key)
    ]

    if len(matching_keys) == 1:
        logger.info(f"Sparse matrix loaded: {matching_keys[0]}")
        return matrix_data[matching_keys[0]].item()
    elif len(matching_keys) > 1:
        raise FileExistsError(
            f"Multiple matching sparse matrices found: {matching_keys}"
        )
    else:
        available_keys = list(matrix_data.keys())
        raise KeyError(
            f"No matching sparse matrix found for set '{set_name}'. "
            f"Available keys: {available_keys}"
        )


def _load_raw_image(dataset_id: DatasetId):
    set_name = dataset_id[0]
    directory_path = Config2.IMAGES_DIR

    pattern = re.compile(
        rf"W-\d+-\d+-\d+_{re.escape(set_name)}\.(tif|png|jpg)", re.IGNORECASE
    )

    file_path = find_file_with_pattern(directory_path, pattern, f"image for {set_name}")
    if file_path is None:
        logger.warning("Background image is None.")
        return None

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    logger.info(f"Image file loaded: {file_path}")
    if image is None:
        err_msg = f"Failed to load image from file: {file_path}"
        logger.error(err_msg)
        return None

    return image
