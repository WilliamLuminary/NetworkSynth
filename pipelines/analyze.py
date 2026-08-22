import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import os
from collections import deque
from typing import Dict

from configs import AnaConfig, DatasetId
from handlers import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    RunAgent,
    attach_run_log,
    create_run_paths,
    write_manifest,
)

logger = logging.getLogger(__name__)


def find_pkl_containers(base_dir: str, max_depth: int = 3) -> Dict[str, str]:
    logger.info(
        f"Searching for container directories in {base_dir} up to depth {max_depth}."
    )
    containers = {}
    queue = deque([(base_dir, 0, "")])  # (path, depth, rel_path)

    while queue:
        current_dir, depth, rel_path = queue.popleft()

        if os.path.basename(current_dir).startswith((".", "__")):
            continue

        has_network_dirs = any(
            entry in ("synthetic", "origin", "original")
            for entry in os.listdir(current_dir)
        )

        if has_network_dirs:
            containers[rel_path] = current_dir
            logger.info(f"Found container directory: {current_dir}")
            continue

        if depth < max_depth:
            for entry in os.listdir(current_dir):
                entry_path = os.path.join(current_dir, entry)
                if os.path.isdir(entry_path):
                    new_rel = os.path.join(rel_path, entry) if rel_path else entry
                    queue.append((entry_path, depth + 1, new_rel))

    logger.info(f"Found {len(containers)} container directories.")
    return containers


def run_for_container(name: str, path: str, config, run_paths) -> None:
    # The container is read, never written to: results go under this run's own
    # root like every other mode, which is what lets a caller find them from
    # the manifest.  A container found at the search root has no relative name.
    dataset_id = DatasetId(*(name.split(os.sep) if name else ("analysis",)))
    logger.info(f"Analysing {path} as {dataset_id}")
    data_agent = RunAgent(
        config, networks_path=path, run_paths=run_paths, dataset_id=dataset_id
    )
    data_agent.prepare_data()
    data_agent.multifractal_analysis()
    data_agent.save("analysis_data")
    data_agent.save("analysis_figure")


def main(config_cls=AnaConfig):
    config_cls.initialize()
    run_paths = create_run_paths(config_cls)
    attach_run_log(run_paths.root, config_cls.RUN_ID)
    status, error = STATUS_OK, None
    try:
        containers = find_pkl_containers(config_cls.NETWORKS_DATA_PATH)
        assert containers, (
            f"no directory holding synthetic/ and original/ found under "
            f"{config_cls.NETWORKS_DATA_PATH}"
        )
        for name, path in containers.items():
            run_for_container(name, path, config_cls, run_paths)
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
