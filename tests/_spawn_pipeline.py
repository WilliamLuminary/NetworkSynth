# tests/_spawn_pipeline.py
"""Run one pipeline under a 'spawn' start method.  Executed as a subprocess.

Not named ``test_*`` on purpose: pytest must not collect it.  It is launched by
``test_spawn_safety.py`` in a separate interpreter, because
``set_start_method`` is process-global and would affect the whole test session.

Under ``spawn`` — the default on macOS and Windows — a child re-imports every
module fresh, so it sees an unmutated ``BaseConfig``.  Anything that still
relied on inheriting the parent's class state fails here.
"""

import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    mp.set_start_method("spawn", force=True)

    from configs.generate_mode.config_sample import SampleConfig
    from handlers import create_run_paths
    from pipelines.generate import run_for_dataset

    out_dir = tempfile.mkdtemp(prefix="spawn_safety_")
    config = type("SpawnSampleConfig", (SampleConfig,), {})
    config.BASE_OUTPUT_PATH = out_dir
    config.ERROR_CHECKER = "none"
    config.MAX_ATTEMPTS = 2
    config.FULL_ANALYSIS = False
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
