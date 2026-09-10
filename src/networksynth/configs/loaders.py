# SPDX-License-Identifier: GPL-3.0-or-later
"""How a dataset is read off disk, shared by every config.

A config names where its files are; this module knows how to read them. The
network is always reduced to its largest connected component here, because the
attribute and multifractal analyses take all-pairs distances, which are infinite
across components.

Imports from networksynth.graphs are deferred: graphs imports configs.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from numpy import ndarray

    from networksynth.graphs.synth_graph import SynthGraph

logger = logging.getLogger(__name__)


class SpecError(ValueError):
    pass


EDGE_SUFFIX = "_edgelist.csv"
POSITIONS_SUFFIX = "_positions.csv"
# What StructuralGT's exporter writes. Its columns are already the ones
# read_graph_csv accepts, so only the file names differ, and its positions are
# the skeleton's (row, col) under headers x and y.
SGT_EDGE_SUFFIX = "_EdgeList.csv"
SGT_POSITIONS_SUFFIX = "_NodePositions.csv"
IMAGE_SUFFIX = "_image.tif"
MATRIX_SUFFIX = "_adjacency.npy"
NPY_POSITIONS_SUFFIX = "_positions.npy"
# The older names the sample data under data/input still carries.
OLD_MATRIX_SUFFIX = "_mat.npy"
OLD_NPY_POSITIONS_SUFFIX = "_pos.npy"
GRAPHML_SUFFIX = "_network.graphml"
GRAPHML_GZ_SUFFIX = "_network.graphml.gz"

# A dataset is named by the file that leads its form, with the partners that
# have to sit beside it.
DIRECTORY_FORMS = (
    (EDGE_SUFFIX, (POSITIONS_SUFFIX,)),
    (SGT_EDGE_SUFFIX, (SGT_POSITIONS_SUFFIX,)),
    (MATRIX_SUFFIX, (NPY_POSITIONS_SUFFIX,)),
    (OLD_MATRIX_SUFFIX, (OLD_NPY_POSITIONS_SUFFIX,)),
    (GRAPHML_SUFFIX, ()),
    (GRAPHML_GZ_SUFFIX, ()),
)


def discover_datasets(directory: str) -> list:
    if not os.path.isdir(directory):
        raise SpecError(f"not a directory: {directory}")

    found = {}
    for entry in sorted(os.listdir(directory)):
        for lead, partners in DIRECTORY_FORMS:
            if not entry.endswith(lead):
                continue
            name = entry[: -len(lead)]
            if name in found:
                raise SpecError(
                    f"{directory}: '{name}' is named as two datasets at once "
                    f"({found[name]} and {lead}). Rename one of them."
                )
            for partner in partners:
                beside = os.path.join(directory, f"{name}{partner}")
                if not os.path.exists(beside):
                    raise SpecError(
                        f"{os.path.join(directory, entry)} has no {partner} "
                        f"file beside it ({beside})"
                    )
            found[name] = lead

    if not found:
        forms = ", ".join(f"*{lead}" for lead, _ in DIRECTORY_FORMS)
        raise SpecError(f"no {forms} file found in {directory}")
    return sorted(found)


def load_csv_pair(edge_list_path: str, positions_path: str) -> SynthGraph:
    from networksynth.graphs import read_graph_csv

    graph = read_graph_csv(edge_list_path, positions_path)
    if positions_path.endswith(SGT_POSITIONS_SUFFIX):
        from networksynth.utils import transpose_positions

        transpose_positions(graph)
    return graph.largest_connected_component()


def load_npy_pair(positions_path: str, adjacency_path: str) -> SynthGraph:
    import numpy as np

    from networksynth.graphs.synth_graph import SynthGraph
    from networksynth.utils import transpose_positions

    positions = np.load(positions_path, allow_pickle=True)
    matrix = np.load(adjacency_path, allow_pickle=True).item()
    graph = SynthGraph.from_sparse_matrix(positions, matrix)
    transpose_positions(graph)
    return graph.largest_connected_component()


def load_network_file(path: str) -> SynthGraph:
    from networksynth.graphs import load_graphs

    graphs = load_graphs(path)
    if len(graphs) > 1:
        logger.warning(
            f"{path} holds {len(graphs)} networks; reading the first. Point at "
            "a single-network file to choose a different one."
        )
    return graphs[0]


def load_network(directory: str, name: str) -> Tuple[SynthGraph, str]:
    """One dataset out of a directory, by the form its name is in.

    Returns the graph and the suffix that identified it, so a caller can tell a
    StructuralGT export from the rest without listing the directory again.
    """
    # A multi-level DatasetId arrives as "a/b", meaning subdirectory a, prefix b.
    directory = os.path.join(directory, os.path.dirname(name))
    name = os.path.basename(name)
    path = os.path.join(directory, name)
    present = set(os.listdir(directory))
    for edges, positions in (
        (EDGE_SUFFIX, POSITIONS_SUFFIX),
        (SGT_EDGE_SUFFIX, SGT_POSITIONS_SUFFIX),
    ):
        if f"{name}{edges}" in present:
            return load_csv_pair(f"{path}{edges}", f"{path}{positions}"), edges
    for matrix, positions in (
        (MATRIX_SUFFIX, NPY_POSITIONS_SUFFIX),
        (OLD_MATRIX_SUFFIX, OLD_NPY_POSITIONS_SUFFIX),
    ):
        if f"{name}{matrix}" in present:
            return load_npy_pair(f"{path}{positions}", f"{path}{matrix}"), matrix
    if f"{name}{GRAPHML_GZ_SUFFIX}" in present:
        return load_network_file(f"{path}{GRAPHML_GZ_SUFFIX}"), GRAPHML_GZ_SUFFIX
    return load_network_file(f"{path}{GRAPHML_SUFFIX}"), GRAPHML_SUFFIX


def load_image(
    directory: str,
    name: str,
    frame_size: Optional[Tuple[int, int]] = None,
    suffix: str = IMAGE_SUFFIX,
) -> Optional[ndarray]:
    """The dataset's background, or None when it has none; images are optional."""
    path = os.path.join(directory, f"{name}{suffix}")
    if not os.path.exists(path):
        logger.warning(f"No image at {path}.")
        return None

    import cv2

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning(f"Image could not be read: {path}")
        return None
    if frame_size is None:
        return image

    from networksynth.utils import resize_image

    return resize_image(image, frame_size)
