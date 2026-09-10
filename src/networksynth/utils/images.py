# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple

from numpy import ndarray


def resize_image(image: ndarray, target_size: Tuple[int, int]) -> ndarray:
    height, width = image.shape[:2]
    scaling_factor = max(target_size) / max(height, width)
    new_height, new_width = (
        round(height * scaling_factor),
        round(width * scaling_factor),
    )
    import cv2

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
