# scripts/run_hybrid_100x100.py
"""
Production run: generate a ~100x100 network using the two-phase hybrid pipeline.

Centers are placed randomly across the whiteboard with a minimum pairwise
distance (default 1.5× frame size), replacing the old regular grid layout.

Phase 1 — Parallel tile generation:
    Each center spawns an independent 1×1 seed tile in its own process,
    quality-checked via multifractal error analysis and retried on failure.

Phase 2 — Assembly & gap-filling (single-process, multi-threaded):
    All tiles are placed at their random center positions on the shared
    whiteboard.  BFS continues from frontier nodes to fill gaps and merge
    adjacent components.

Usage (from project root):
    python scripts/run_hybrid_100x100.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
import time

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "hybrid_100x100.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
root_logger.addHandler(fh)

sh = logging.StreamHandler(stream=open(sys.stdout.fileno(), "w", buffering=1))
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
root_logger.addHandler(sh)

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Hybrid 100×100 production run")
logger.info("=" * 60)

from configs import BaseConfig
from configs.hybrid_mode import HybridConfig
from handlers import Saver
from pipelines.hybrid import run_hybrid_for_dataset

HybridConfig.NUM_CENTERS = 2000
HybridConfig.initialize()

logger.info(f"Whiteboard scale: {BaseConfig.HYBRID_ROWS}×{BaseConfig.HYBRID_COLS}")
logger.info(f"Num centers: {BaseConfig.NUM_CENTERS}")
logger.info(f"Frame: {BaseConfig.SYNTHETIC_FRAME_SIZE}")
min_dist_factor = getattr(BaseConfig, "MIN_CENTER_DISTANCE_FACTOR", 1.5)
logger.info(f"Min center distance factor: {min_dist_factor}")
logger.info(f"Phase 2 max rounds: {BaseConfig.PHASE2_MAX_ROUNDS}")
logger.info(f"CPU count: {os.cpu_count()}")
logger.info(f"Max workers: {BaseConfig.get_max_workers()}")

t_start = time.time()

try:
    Saver.initialize()
    for dataset_id in BaseConfig.get_datasets():
        run_hybrid_for_dataset(dataset_id)
except KeyboardInterrupt:
    logger.critical("Interrupted by user")
except Exception:
    logger.exception("Fatal error")
finally:
    elapsed = time.time() - t_start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
