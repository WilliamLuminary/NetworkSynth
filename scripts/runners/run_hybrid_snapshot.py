# scripts/runners/run_hybrid_snapshot.py
"""
Hybrid snapshot run: generate a hybrid network for sample_A and capture
Phase 2 snapshots showing how tiles merge round by round.

Usage (from project root):
    python scripts/runners/run_hybrid_snapshot.py
    python scripts/runners/run_hybrid_snapshot.py --scale 5    # 5x5 (faster)
    python scripts/runners/run_hybrid_snapshot.py --scale 100  # 100x100 (full)
    python scripts/runners/run_hybrid_snapshot.py --interval 5 # snapshot every 5 rounds
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time

import fire

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


def main(scale: int = 10, interval: int = 1, config: str = "A"):
    """Run hybrid snapshot generation.

    Args:
        scale: Scale factor (NxN). Default 10 for fast iteration.
        interval: Snapshot every N Phase 2 rounds. Default 1 (every round).
        config: Which sample dataset (A/B/C/D). Default A.
    """
    logger.info("=" * 60)
    logger.info(f"Hybrid snapshot run — sample_{config} @ {scale}x{scale}")
    logger.info("=" * 60)

    from configs.enums import DatasetId
    from configs.hybrid_mode import HybridConfig
    from handlers import create_run_paths
    from pipelines.hybrid import run_hybrid_for_dataset

    HybridConfig.TARGET_SCALE = (scale, scale)
    HybridConfig.SNAPSHOT_INTERVAL = interval
    HybridConfig.initialize()

    dataset_id = DatasetId(f"sample_{config}")

    logger.info(f"Target scale: {HybridConfig.TARGET_SCALE}")
    logger.info(f"Frame: {HybridConfig.SYNTHETIC_FRAME_SIZE}")
    logger.info(f"Phase 2 max rounds: {HybridConfig.PHASE2_MAX_ROUNDS}")
    logger.info(f"Snapshot interval: every {interval} round(s)")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"Max workers: {HybridConfig.get_max_workers()}")

    t_start = time.time()

    try:
        run_paths = create_run_paths(HybridConfig)
        run_hybrid_for_dataset(dataset_id, HybridConfig, run_paths)
    except KeyboardInterrupt:
        logger.critical("Interrupted by user")
    except Exception:
        logger.exception("Run failed")
    finally:
        elapsed = time.time() - t_start
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
