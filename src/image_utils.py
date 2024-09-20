import cv2
import os
import numpy as np

def load_image_file(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from file: {image_file}")
    return image

def resize_image_to_fit_positions(image, target_size=510):
    original_height, original_width = image.shape
    scaling_factor = target_size / max(original_width, original_height)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor

