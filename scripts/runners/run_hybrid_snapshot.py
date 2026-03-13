# scripts/runners/run_hybrid_snapshot.py
"""
Hybrid snapshot run: generate a hybrid network for sample_A and capture
Phase 2 snapshots showing how tiles merge round by round.

Usage (from project root):
    python scripts/runners/run_hybrid_snapshot.py
    python scripts/runners/run_hybrid_snapshot.py --scale 5   # 5x5 (faster)
    python scripts/runners/run_hybrid_snapshot.py --scale 100  # 100x100 (full)
    python scripts/runners/run_hybrid_snapshot.py --interval 5  # snapshot every 5 rounds
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import logging
import time

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "hybrid_snapshot.log")

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

parser = argparse.ArgumentParser(description="Hybrid snapshot generation")
parser.add_argument(
    "--scale",
    type=int,
    default=10,
    help="Scale factor (NxN). Default 10 for fast iteration.",
)
parser.add_argument(
    "--interval",
    type=int,
    default=1,
    help="Snapshot every N Phase 2 rounds. Default 1 (every round).",
)
parser.add_argument(
    "--config",
    choices=["A", "B", "C", "D"],
    default="A",
    help="Which sample dataset. Default A.",
)
args = parser.parse_args()

logger.info("=" * 60)
logger.info(f"Hybrid snapshot run — sample_{args.config} @ {args.scale}x{args.scale}")
logger.info("=" * 60)

from configs import BaseConfig
from configs.enums import DatasetId
from configs.hybrid_mode import HybridConfig
from handlers import Saver
from pipelines.hybrid import run_hybrid_for_dataset

HybridConfig.TARGET_SCALE = (args.scale, args.scale)
HybridConfig.SNAPSHOT_INTERVAL = args.interval
HybridConfig.initialize()

dataset_id = DatasetId(f"sample_{args.config}")

logger.info(f"Target scale: {BaseConfig.TARGET_SCALE}")
logger.info(f"Frame: {BaseConfig.SYNTHETIC_FRAME_SIZE}")
logger.info(f"Phase 2 max rounds: {BaseConfig.PHASE2_MAX_ROUNDS}")
logger.info(f"Snapshot interval: every {args.interval} round(s)")
logger.info(f"CPU count: {os.cpu_count()}")
logger.info(f"Max workers: {BaseConfig.get_max_workers()}")

t_start = time.time()

try:
    Saver.initialize()
    run_hybrid_for_dataset(dataset_id)
except KeyboardInterrupt:
    logger.critical("Interrupted by user")
except Exception:
    logger.exception("Run failed")
finally:
    elapsed = time.time() - t_start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
