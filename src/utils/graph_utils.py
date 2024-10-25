import numpy as np

def adjust_positions(positions, flip_axes=False, invert_y=False, scale_factor=None):
    positions_array = np.array([positions[node] for node in positions])

    if flip_axes:
        positions_array[:, [1, 0]] = positions_array[:, [0, 1]]

    if invert_y:
        positions_array[:, 1] = scale_factor - positions_array[:, 1] if scale_factor else -positions_array[:, 1]

    adjusted_positions = {node: pos for node, pos in zip(positions.keys(), positions_array)}
    return adjusted_positions