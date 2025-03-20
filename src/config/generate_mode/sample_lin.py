# src/config/generate_mode/sample_lin.py
import logging
import os
import pickle

import numpy as np

from .._utils import _find_file_with_pattern
from ..base_config import BaseConfig
from ..enums import Resolution, SetName

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    SETS = [SetName.IgraphSample1]  # TODO: Add more sets in the src/config/enum.py
    RESOLUTIONS = [Resolution.NA]

    DEFAULT_FRAME_SIZE = (1887 // 4, 2048 // 4)  # TODO: Set the size manually
    CLOSED_NODES_FACTOR = 1.2  # You may need to sweep
    CLOSED_EDGES_FACTOR = .8  # You may need to sweep

    SYNTHETIC_GRAPH_NUMBER = 0  # Please set this yourself
    SYNTHETIC_NETWORK_NUMBER = 0  # Please set this yourself

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'linlin_input')

    # BASE_INPUT_PATH
    # ├── POSITION_DATA_DIR
    # ├── ADJ_MATRIX_DATA_DIR
    # └── IMAGES_DIR

    @classmethod
    def initialize(cls):
        super().initialize()
        cls.ORIGINAL_NETWORK_FUNC = cls.load_original_network
        cls.ORIGINAL_IMAGE_FUNC = cls.load_original_image
        cls._inject_dependencies()

    @staticmethod
    def load_original_network(set_name, _):
        positions = _load_positions(set_name)
        edge_list = _load_edge_list(set_name)
        from utils import build_graph
        original_network = build_graph(positions, edge_list, arg_type='edge_list')
        return original_network

    @staticmethod
    def load_original_image(*_):
        # image = _load_raw_image(_)
        # image = _trim_cv2_image(image)
        # image = _resize_cv2_image(image)
        return None


def _load_positions(*_):
    file_path = os.path.join(SampleConfig.BASE_INPUT_PATH, f"sample_lin_pos.npy")
    logger.info(f"Loading positions from {file_path}")
    return np.load(file_path, allow_pickle=True)


def _load_edge_list(*_):
    file_path = os.path.join(SampleConfig.BASE_INPUT_PATH, "sample_lin_mat.npy")
    logger.info(f"Loading edge list from {file_path}")
    # Assuming the file contains an edge list in which the first two columns represent the source and target.
    return np.load(file_path, allow_pickle=True)[:, :2]


def _load_igraph(set_name):
    """
    Load the igraph object from the specified directory.
    :return: The igraph object.
    """
    file_pattern = rf"{set_name}_igraph\.pkl"
    file_path = _find_file_with_pattern(SampleConfig.BASE_INPUT_PATH, file_pattern, details='igraph')
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)

    import networkx as nx
    assert isinstance(graph, nx.Graph)
    return graph


def _load_raw_image(*_):
    """
    Load the raw image from the specified directory.
    :return:
    """
    return None
