from typing import Optional, Tuple

from numpy import ndarray


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


def _resize_cv2_image(image: ndarray, target_size: Tuple[int, int]) -> ndarray:
    """Resize *image* so its longest side matches *target_size*.

    The size is required: reading it from ``BaseConfig`` only worked while
    configs published their values there, and silently used the wrong frame for
    any config that had not.
    """
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
    top, bottom, left, right = trim
    if not any([top, bottom, left, right]):
        return image
    h, w = image.shape[:2]
    return image[top : h - bottom if bottom else h, left : w - right if right else w]


def _transpose_network_pos(network) -> None:
    pos = network.positions()
    pos[:, [0, 1]] = pos[:, [1, 0]]
