"""Image operations on input data.

The cv2 work that reading an input involves.  Rendering lives in
:mod:`utils.plotting`; this is the other half — preparing the background image a
config hands to the pipeline.
"""

from typing import Tuple

from numpy import ndarray


def resize_image(image: ndarray, target_size: Tuple[int, int]) -> ndarray:
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

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


def trim_image(
    image: ndarray, trim: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> ndarray:
    """Crop (top, bottom, left, right) pixels off *image*."""
    top, bottom, left, right = trim
    if not any([top, bottom, left, right]):
        return image
    h, w = image.shape[:2]
    return image[top : h - bottom if bottom else h, left : w - right if right else w]
