# SPDX-License-Identifier: GPL-3.0-or-later
import logging
import sys
from pathlib import Path

from networksynth.configs.loader import load_config
from networksynth.handlers import configure_console
from networksynth.pipelines import PIPELINES, load_pipeline

logger = logging.getLogger("run")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_ARGS = 2
EXIT_INTERRUPTED = 130

_USAGE = f"""usage: networksynth-cli <path/to/config.py>

The config's MODE picks the pipeline: {", ".join(sorted(PIPELINES))}.

Available configs:
"""


def _available() -> str:
    root = Path(__file__).resolve().parent / "configs"
    configs = sorted(root.glob("*_mode/config_*.py"))
    return "\n".join(f"  {path}" for path in configs)


def main(argv=None) -> int:
    configure_console()
    argv = sys.argv if argv is None else argv

    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(_USAGE + _available())
        return EXIT_BAD_ARGS

    try:
        config = load_config(argv[1])
        pipeline = load_pipeline(config.MODE)
    except (FileNotFoundError, ValueError, AttributeError) as exc:
        logger.error(f"{exc}")
        return EXIT_BAD_ARGS

    logger.info(f"Running {config.__name__} ({config.MODE})")
    try:
        pipeline.main(config_cls=config)
    except KeyboardInterrupt:
        logger.critical("Interrupted — exiting.")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("Run failed")
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
