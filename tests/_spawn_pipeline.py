# SPDX-License-Identifier: GPL-3.0-or-later
import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    mp.set_start_method("spawn", force=True)

    from networksynth.configs.generate_mode.config_sample import SampleConfig
    from networksynth.handlers import create_run_paths
    from networksynth.pipelines.generate import run_for_dataset

    out_dir = tempfile.mkdtemp(prefix="spawn_safety_")
    config = type("SpawnSampleConfig", (SampleConfig,), {})
    config.BASE_OUTPUT_PATH = out_dir
    config.ERROR_CHECKER = "none"
    config.MAX_ATTEMPTS = 2
    config.LOG_MEMORY = False
    config.SEED = 4321
    config.SYNTHETIC_NETWORK_NUMBER = 2
    config.SYNTHETIC_GRAPH_NUMBER = 1
    config.SNAPSHOT_INTERVAL = 0
    config.initialize()

    run_for_dataset(config.get_datasets()[0], config, create_run_paths(config))

    written = sum(len(names) for _, _, names in os.walk(out_dir))
    print(f"files_written={written}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
