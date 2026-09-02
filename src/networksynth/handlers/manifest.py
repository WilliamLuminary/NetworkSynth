# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_IMAGE_SUFFIXES = (".webp", ".png", ".svg", ".jpg", ".jpeg", ".tif", ".tiff")


def _classify(relative_path: str) -> str:
    lowered = relative_path.lower()
    parts = lowered.split("/")

    if lowered.endswith("_edgelist.csv"):
        return "edge_lists"
    if lowered.endswith("_positions.csv") or lowered.endswith("_positions.npy"):
        return "positions"
    if "snapshots" in parts:
        return "snapshots"
    if "analysis" in lowered:
        return "analysis"
    if lowered.endswith(_IMAGE_SUFFIXES):
        return "previews"
    if lowered.endswith((".graphml", ".graphml.gz", ".graphmlz")):
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
