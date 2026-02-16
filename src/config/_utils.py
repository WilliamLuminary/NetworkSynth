from typing import Optional, Tuple

from networkx import Graph
from numpy import array, float64, ndarray

from config import BaseConfig


def _find_file_with_pattern(
    directory_path, pattern, details="", must_exist=True
) -> Optional[str]:
    import os

    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Directory not found: {directory_path}. Regex pattern: {pattern}"
        )

    import re

    regex_pattern = re.compile(pattern, re.IGNORECASE)
    files_in_directory = os.listdir(directory_path)
    matched_files = [
        file_name for file_name in files_in_directory if regex_pattern.search(file_name)
    ]

    if len(matched_files) == 1:
        file_path = os.path.join(directory_path, matched_files[0])
        return file_path
    elif len(matched_files) > 1:
        raise FileExistsError(
            f"Multiple {details} files found: {matched_files}. Regex pattern: {pattern}"
        )
    else:
        assert (
            not must_exist
        ), f"No {details} file found in {directory_path}. Regex pattern: {pattern}"
        return None


def _resize_cv2_image(image: ndarray, target_size: Tuple[int, int] = None) -> ndarray:
    """Resize image to target size or FRAME_SIZE if not specified."""
    target_size = target_size or BaseConfig.FRAME_SIZE
    height, width = image.shape[:2]
    scaling_factor = max(target_size) / max(height, width)
    new_height, new_width = round(height * scaling_factor), round(
        width * scaling_factor
    )
    import cv2

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image


def _trim_cv2_image(
    image: ndarray, trim: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> ndarray:
    """Trim image edges. trim = (top, bottom, left, right)."""
    top, bottom, left, right = trim
    if not any([top, bottom, left, right]):
        return image
    h, w = image.shape[:2]
    return image[top : h - bottom if bottom else h, left : w - right if right else w]


def _transpose_network_pos(network: Graph) -> None:
    for node, d in network.nodes(data=True):
        p = array(d.get("pos", [0, 0]), dtype=float64)
        d["pos"] = array([p[1], p[0]])
