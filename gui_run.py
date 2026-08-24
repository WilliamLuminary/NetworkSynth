import logging
import os
import sys

from configs.gui_config import GuiConfig, SpecError
from handlers import configure_console
from pipelines import load_pipeline

logger = logging.getLogger("gui_run")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_SPEC = 2
EXIT_INTERRUPTED = 130

#: Only the modes the GUI actually offers.  A subset of ``pipelines.PIPELINES``,
#: which is the full table: add to this deliberately, since each entry needs its
#: loader slots and its input shapes checked, not just a name.
_GUI_MODES = frozenset({"generate", "hybrid", "sweep", "compare"})


def _resolve_spec_path(argv) -> str:
    if len(argv) > 1:
        return argv[1]
    from_env = os.environ.get("NETWORKSYNTH_RUN_SPEC")
    if from_env:
        return from_env
    raise SpecError(
        "no run-spec given. Pass a path as the first argument or set "
        "NETWORKSYNTH_RUN_SPEC."
    )


def main(argv=None) -> int:
    configure_console()
    argv = sys.argv if argv is None else argv

    try:
        spec_path = _resolve_spec_path(argv)
        config = GuiConfig.from_spec(spec_path)
        if config.MODE not in _GUI_MODES:
            raise SpecError(
                f"mode {config.MODE!r} is not available; "
                f"this build supports: {sorted(_GUI_MODES)}"
            )
    except SpecError as exc:
        logger.error(f"Bad run-spec: {exc}")
        return EXIT_BAD_SPEC

    pipeline = load_pipeline(config.MODE)

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
