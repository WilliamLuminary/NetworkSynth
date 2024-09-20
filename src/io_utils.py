import os
import numpy as np

def load_positions(set_name, resolution, base_path):
    positions_path = os.path.join(base_path, 'position')
    positions_file = os.path.join(positions_path, f'{set_name}_{resolution}_postion.npy')
    return np.load(positions_file, allow_pickle=True)

def load_sparse_matrix(set_name, resolution, base_path):
    sparse_matrices_path = os.path.join(base_path, 'sparse_matrices')
    matrix_file = os.path.join(sparse_matrices_path, f'sparse_matrices_{resolution}.npz')
    matrix_data = np.load(matrix_file, allow_pickle=True)
    return matrix_data[set_name].item()