import os
import re
import warnings

import cv2
import numpy as np

from src.utils.debug_utils import debugging


@debugging
def find_file_with_pattern(directory_path, pattern, details=''):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files_in_directory = os.listdir(directory_path)

    details_str = f' {details} ' if details else ' '
    matched_files = [file_name for file_name in files_in_directory if pattern.search(file_name)]

    if len(matched_files) == 1:
        file_path = os.path.join(directory_path, matched_files[0])
        print(f'{details_str} {file_path}', debug=True)
        return file_path
    elif len(matched_files) > 1:
        raise FileExistsError(f"Multiple{details_str}files found matching the pattern: {matched_files}.")
    else:
        raise FileNotFoundError(f"No{details_str}file found in {directory_path}. Available files: {files_in_directory}")


@debugging
def load_positions(set_name, resolution, base_path='/content/drive/MyDrive/vis'):
    positions_path = os.path.join(base_path, 'position')
    pattern = re.compile(rf"{re.escape(set_name)}_{re.escape(resolution)}.*(pos|postion)\.npy", re.IGNORECASE)
    return np.load(find_file_with_pattern(positions_path, pattern, details=f'position for {set_name}'),
                   allow_pickle=True)


def load_sparse_matrix(resolution, base_path='/content/drive/MyDrive/vis', set_name=None):
    sparse_matrices_path = os.path.join(base_path, 'sparse_matrices')
    pattern = re.compile(rf"sparse_matrices_{re.escape(resolution)}.*\.npz", re.IGNORECASE)
    matrix_file = find_file_with_pattern(sparse_matrices_path, pattern, details='sparse matrix')
    matrix_data = np.load(matrix_file, allow_pickle=True)

    if set_name == 'C':
        if 'C1' in matrix_data:
            warnings.warn("Set name 'C' uses 'C1' by default.")
            set_name = 'C1'
        elif 'C' in matrix_data:
            warnings.warn("Using 'C' since 'C1' is not available.")
            set_name = 'C'
        else:
            available_keys = list(matrix_data.keys())
            raise KeyError(f"Neither 'C' nor 'C1' is found in the matrix file. Available sets: {available_keys}")

    if set_name and set_name not in matrix_data:
        available_keys = list(matrix_data.keys())
        raise KeyError(f"Matrix data for set '{set_name}' not found in {matrix_file}. Available sets: {available_keys}")

    return matrix_data[set_name].item()


def find_image_file(set_name, resolution, base_path='/content/drive/MyDrive/vis/Original Graphs'):
    directory_path = os.path.join(base_path, set_name)
    if set_name == 'B':
        directory_path = os.path.join(directory_path, '1811')
        warnings.warn("For set B, using the 1811 subdirectory for images.")
    pattern = re.compile(rf"\b{re.escape(resolution)}\b.*\.(tif|png|jpg)", re.IGNORECASE)
    return find_file_with_pattern(directory_path, pattern, details=f'image file for {set_name}')


def load_image_file(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from file: {image_file}")
    return image
