# src/config/generate_mode/_utils.py
from typing import Tuple

import cv2
import networkx as nx
import numpy as np

from config import BaseConfig


def _resize_cv2_image(image: np.ndarray) -> np.ndarray:
    frame_range = BaseConfig.DEFAULT_FRAME_SIZE
    height, width = image.shape[:2]
    scaling_factor = max(frame_range) / max(height, width)
    new_height, new_width = round(height * scaling_factor), round(width * scaling_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def _trim_cv2_image(image: np.ndarray, trim: Tuple[int, int, int, int] = None) -> np.ndarray:
    trim = trim or BaseConfig.TRIM_SIZE
    top, bottom, left, right = trim
    image = image[top:image.shape[0] - bottom, left:image.shape[1] - right]
    return image


def _transpose_network_pos(network: nx.Graph) -> None:
    for node, d in network.nodes(data=True):
        p = np.array(d.get('pos', [0, 0]), dtype=np.float64)
        d['pos'] = np.array([p[1], p[0]])
