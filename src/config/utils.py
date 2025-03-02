import os
from typing import Optional, Tuple

import cv2
import networkx as nx
import numpy as np

from config import BaseConfig
from config.base_config import logger


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


def _transpose_coordinates(network: nx.Graph) -> None:
    for node, d in network.nodes(data=True):
        p = np.array(d.get('pos', [0, 0]), dtype=np.float64)
        d['pos'] = np.array([p[1], p[0]])


def _find_file_with_pattern(directory_path, pattern, details='', must_exist=True) -> Optional[str]:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files_in_directory = os.listdir(directory_path)
    matched_files = [file_name for file_name in files_in_directory if pattern.search(file_name)]

    if len(matched_files) == 1:
        file_path = os.path.join(directory_path, matched_files[0])
        logger.info(f"{details} file found: {file_path}")
        return file_path
    elif len(matched_files) > 1:
        raise FileExistsError(f"Multiple {details} files found: {matched_files}.")
    else:
        if must_exist:
            raise FileNotFoundError(f"No {details} file found in {directory_path}.")
        else:
            logger.warning(f"No {details} file found in {directory_path}.")
            return None
