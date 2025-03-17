# src/config/generate_mode/sample.py
import logging
import os

from .._utils import _transpose_network_pos
from ..base_config import BaseConfig
from ..enums import Resolution, SetName

logger = logging.getLogger(__name__)


class SampleConfig(BaseConfig):
    SETS = [SetName.NA]                                                                 # TODO: Add more sets in the src/config/enum.py
    RESOLUTIONS = [Resolution.NA]

    DEFAULT_FRAME_SIZE = (1887 // 4, 2048 // 4)                                         # TODO: Set the size manually
    CLOSED_NODES_FACTOR = 1.2                                                           # You may need to sweep
    CLOSED_EDGES_FACTOR = .8                                                            # You may need to sweep

    SYNTHETIC_GRAPH_NUMBER = 0                                                          # Please set this yourself
    SYNTHETIC_NETWORK_NUMBER = 0                                                        # Please set this yourself

    MAX_ATTEMPTS = 10
    ERROR_TOLERANCE = .15

    MEASURE_WEIGHTED = False

    BASE_INPUT_PATH = os.path.join(BaseConfig.BASE_INPUT_PATH, 'input', 'customized')   # TODO: Set accordingly
    IGRAPH = os.path.join(BASE_INPUT_PATH, 'graph')                                     # TODO: Set accordingly
    IMAGES_DIR = os.path.join(BASE_INPUT_PATH, 'images')                                # TODO: Set accordingly

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
    def load_original_network(_):
        import igraph
        # noinspection PyUnresolvedReferences
        original_network = igraph.to_networkx()
        _transpose_network_pos(original_network)
        return original_network

    @staticmethod
    def load_original_image(_):
        image = _load_raw_image(_)
        # image = _trim_cv2_image(image)
        # image = _resize_cv2_image(image)
        return image


def _load_igraph(_):
    """
    Load the igraph object from the specified directory.
    :return: The igraph object.
    """
    raise NotImplementedError("Please implement this function.") # TODO: Load the igraph object
    # igraph = None
    # return igraph


def _load_raw_image(_):
    """
    Load the raw image from the specified directory.
    :return:
    """
    raise NotImplementedError("Please implement this function.") # TODO: Load the raw image
    # image = None
    # return image
