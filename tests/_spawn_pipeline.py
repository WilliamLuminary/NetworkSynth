# SPDX-License-Identifier: GPL-3.0-or-later
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import replace

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)


def main() -> int:
    mp.set_start_method("spawn", force=True)

    from networksynth.handlers import create_run_paths
    from networksynth.pipelines.generate import run_for_dataset
    from tests.fixture_config import FIXTURE

    out_dir = tempfile.mkdtemp(prefix="spawn_safety_")
    config = replace(
        FIXTURE,
        BASE_OUTPUT_PATH=out_dir,
        ERROR_CHECKER="none",
        MAX_ATTEMPTS=2,
        LOG_MEMORY=False,
        SEED=4321,
        SYNTHETIC_NETWORK_NUMBER=2,
        SYNTHETIC_GRAPH_NUMBER=1,
        SNAPSHOT_INTERVAL=0,
    )

    run_for_dataset(config.DATASETS[0], config, create_run_paths(config))

    written = sum(len(names) for _, _, names in os.walk(out_dir))
    print(f"files_written={written}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
