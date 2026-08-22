import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import os
from collections import deque
from typing import Dict

from configs import AnaConfig
from handlers import RunAgent

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


def main(config_cls=AnaConfig):
    config_cls.initialize()
    data_dict = find_pkl_containers(config_cls.NETWORKS_DATA_PATH)
    for name, path in data_dict.items():
        data_agent = RunAgent(config_cls, networks_path=path)
        data_agent.prepare_data()
        data_agent.multifractal_analysis()
        data_agent.save("analysis_data")
        data_agent.save("analysis_figure")


if __name__ == "__main__":
    main()
