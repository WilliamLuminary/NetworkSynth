from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

#: Run outcomes.  ``cancelled`` is distinct from ``failed`` because a GUI
#: should not show an error for a user pressing Cancel.
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_IMAGE_SUFFIXES = (".webp", ".png", ".svg", ".jpg", ".jpeg", ".tif", ".tiff")


def _classify(relative_path: str) -> str:
    """Bucket a file by its location and name, not by which call wrote it."""
    lowered = relative_path.lower()
    parts = lowered.split("/")

    if lowered.endswith("_edgelist.csv"):
        return "edge_lists"
    if lowered.endswith("_positions.csv") or lowered.endswith("_positions.npy"):
        return "positions"
    # Before the image check: a snapshot is an image too, but it belongs to an
    # animation sequence rather than being a final render, and a consumer wants
    # to tell them apart.
    if "snapshots" in parts:
        return "snapshots"
    # Before the image and pickle checks, or the analyse mode's own outputs get
    # filed as previews and networks — a spectrum plot is not a render of a
    # network, and its data pickle is not a network to load.
    if "analysis" in lowered:
        return "analysis"
    if lowered.endswith(_IMAGE_SUFFIXES):
        return "previews"
    if lowered.endswith(".nkbin") or lowered.endswith(".pkl"):
        return "networks"
    return "other"


def collect_outputs(root: str) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == MANIFEST_NAME:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(
                os.sep, "/"
            )
            buckets.setdefault(_classify(rel), []).append(rel)
    for paths in buckets.values():
        paths.sort()
    return buckets


def write_manifest(
    run_paths,
    *,
    status: str,
    error: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> str:
    root = run_paths.root
    outputs = collect_outputs(root)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "status": status,
        "run_root": os.path.abspath(root),
        "outputs": outputs,
        "file_count": sum(len(v) for v in outputs.values()),
    }
    if error:
        payload["error"] = error
    if metrics:
        payload["metrics"] = metrics

    path = os.path.join(root, MANIFEST_NAME)
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info(f"Wrote manifest: {path} (status={status})")
    return path


def read_manifest(root: str) -> dict:
    with open(os.path.join(root, MANIFEST_NAME)) as handle:
        return json.load(handle)
