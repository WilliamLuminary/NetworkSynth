# src/config/generate_mode/sample_igraph.py
import logging
import os
import pickle

from .._utils import _find_file_with_pattern, _transpose_network_pos
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

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'sample_input', 'generate_mode')
    GRAPHS_DIR = os.path.join(BASE_INPUT_PATH, 'graph')
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'images')

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
        graph = _load_igraph(set_name)
        _transpose_network_pos(graph)
        return graph

    @staticmethod
    def load_original_image(*_):
        # image = _load_raw_image(_)
        # image = _trim_cv2_image(image)
        # image = _resize_cv2_image(image)
        return None


def _load_igraph(set_name):
    """
    Load the igraph object from the specified directory.
    :return: The igraph object.
    """
    file_pattern = rf"{set_name}_igraph\.pkl"
    file_path = _find_file_with_pattern(SampleConfig.BASE_INPUT_PATH, file_pattern, details='igraph')
    import numpy
    logger.info(f"Numpy version: {numpy.__version__}")
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    import igraph as ig
    assert isinstance(graph, ig.Graph)
    # noinspection PyUnresolvedReferences
    graph = ig.to_networkx(graph)
    return graph


def _load_raw_image(_):
    """
    Load the raw image from the specified directory.
    :return:
    """
    return None
