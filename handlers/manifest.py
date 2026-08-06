# src/handlers/manifest.py
"""A machine-readable index of what a run produced.

Output lands in a timestamped directory whose name a caller cannot predict
(``<denote>_results_<ts>_<runid>``), and files are written by several different
paths — ``Saver``, the snapshot plot pool, the multifractal figures.  A GUI
launching us as a subprocess should not have to guess at directory names or
extensions, so the run writes ``manifest.json`` at its root saying what exists
and how it finished.

The manifest is produced by walking the run root rather than by having each
writer register itself: that way it also captures files written by the plot
pool, which ``Saver`` never sees.

``manifest_version`` exists so a future change to this shape fails loudly on the
consumer side instead of being misread.  See ``INTEGRATION_PLAN.md``.
"""

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
    if lowered.endswith(_IMAGE_SUFFIXES):
        return "previews"
    if lowered.endswith(".nkbin") or lowered.endswith(".pkl"):
        return "networks"
    if "analysis" in lowered:
        return "analysis"
    return "other"


def collect_outputs(root: str) -> Dict[str, List[str]]:
    """Group every file under *root* into buckets, keyed by relative path.

    Paths use forward slashes regardless of platform so a consumer reading the
    JSON does not have to normalise them.
    """
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
    """Write ``manifest.json`` at the run root and return its path.

    Called in a ``finally`` so a cancelled or failed run still leaves a
    manifest — a caller needs to distinguish "no manifest, we died hard" from
    "manifest says cancelled".
    """
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
    """Read a run's manifest.  Convenience for tests and for a consumer."""
    with open(os.path.join(root, MANIFEST_NAME)) as handle:
        return json.load(handle)
