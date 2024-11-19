import cv2
import numpy as np


def adjust_positions(positions, flip_axes=False, invert_y=False, scale_factor=None):
    positions_array = np.array([positions[node] for node in positions])

    if flip_axes:
        positions_array[:, [1, 0]] = positions_array[:, [0, 1]]

    if invert_y:
        positions_array[:, 1] = scale_factor - positions_array[:, 1] if scale_factor else -positions_array[:, 1]

    adjusted_positions = {node: pos for node, pos in zip(positions.keys(), positions_array)}
    return adjusted_positions


def resize_image_to_fit_positions(image, target_size=510):
    original_height, original_width = image.shape
    scaling_factor = target_size / max(original_width, original_height)

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor
