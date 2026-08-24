import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging

from configs import CompareConfig, DatasetId
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    ComparisonRun,
    attach_run_log,
    create_run_paths,
    write_manifest,
)

logger = logging.getLogger(__name__)


def run_for_dataset(dataset_id: DatasetId, config, run_paths) -> None:
    original, synthetic = config.ORIGINAL_NETWORKS_PATH, config.SYNTHETIC_NETWORKS_PATH
    logger.info(f"Comparing {synthetic} against {original}")
    run = ComparisonRun(config, run_paths, dataset_id, original, synthetic)
    run.analyse()
    run.save_analysis()


def main(config_cls=CompareConfig):
    config_cls.initialize()
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        for dataset_id in config_cls.get_datasets():
            run_for_dataset(dataset_id, config_cls, run_paths)
    except KeyboardInterrupt:
        status = STATUS_CANCELLED
        logger.critical("MAIN PROCESS: Forcing immediate shutdown!")
        raise
    except Exception as exc:
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not config_cls.DISABLE_SAVING:
            write_manifest(run_paths, status=status, error=error)


if __name__ == "__main__":
    main()
