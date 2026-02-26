# scripts/run_hybrid_100x100.py
"""
Production run: generate a 100x100 network using the two-phase hybrid pipeline.

Phase 1 — Parallel tile generation (50x50 = 2,500 independent 1x1 tiles):
    Each tile is generated in its own process, quality-checked via
    multifractal error analysis, and retried if the error exceeds the
    tolerance.  OMP_NUM_THREADS is set to 1 so each worker uses a
    single thread (networkit parallelism disabled for multiprocessing
    safety).

Phase 2 — Assembly & gap-filling (single-process, multi-threaded):
    All tile results are placed on a shared 100x100 whiteboard.
    A synchronized BFS continues from each tile's frontier nodes to
    fill the gaps between tiles and merge adjacent components.

Logs are written to ``data/output/logs/hybrid_100x100.log`` for
real-time monitoring via ``tail -f``.

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

HybridConfig.initialize()

logger.info(f"Grid: {BaseConfig.HYBRID_ROWS}×{BaseConfig.HYBRID_COLS} centers")
logger.info(f"Frame: {BaseConfig.SYNTHETIC_FRAME_SIZE}")
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
