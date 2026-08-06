# src/handlers/run_logging.py
"""Logging setup, as a process concern rather than a config one.

``BaseConfig._setup_logger()`` used to do this, which produced two defects:

* The JSON log went to ``BaseConfig.BASE_OUTPUT_PATH/logs/`` regardless of where
  the active config was writing, so a run told to output somewhere else still
  logged into the repository.  That breaks the integration plan, where the
  run-spec names an ``output_dir`` and StructuralGT's controller tails the log
  inside it.
* A ``_logger_initialized`` latch meant the *first* config to initialize named
  the log file and any later run in the same process got no file handler at all.

Split here into two steps, because they are known at different times:

* :func:`configure_console` — immediately, so early failures are visible.
* :func:`attach_run_log` — once the run's output directory exists, since that is
  where its log belongs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Handlers we installed, so repeat calls replace rather than duplicate them.
_CONSOLE_FLAG = "_networksynth_console"
_RUN_LOG_FLAG = "_networksynth_run_log"

_CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _existing(flag: str) -> Optional[logging.Handler]:
    for handler in logging.getLogger().handlers:
        if getattr(handler, flag, False):
            return handler
    return None


def configure_console(level: int = logging.INFO) -> None:
    """Attach a plain-text console handler to the root logger.

    Idempotent: calling it again adjusts the level rather than adding a second
    handler and double-printing every line.
    """
    root = logging.getLogger()
    root.setLevel(level)

    existing = _existing(_CONSOLE_FLAG)
    if existing is not None:
        existing.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    setattr(handler, _CONSOLE_FLAG, True)
    root.addHandler(handler)


def attach_run_log(
    run_root: str, run_id: str, level: int = logging.INFO
) -> Optional[str]:
    """Write this run's JSON-lines log inside *run_root*.  Returns its path.

    Replaces any previously attached run log, so a second run in the same
    process gets its own file instead of silently continuing to write to the
    first one's — or, as before, getting no file at all.
    """
    from configs.base_config import _JsonFormatter

    root = logging.getLogger()
    root.setLevel(level)

    previous = _existing(_RUN_LOG_FLAG)
    if previous is not None:
        root.removeHandler(previous)
        previous.close()

    if not os.path.isdir(run_root):
        # create_run_paths() does not create the directory when saving is
        # disabled, and a disabled run should leave no trace — including no log.
        return None

    path = os.path.join(run_root, "run.jsonl")
    handler = logging.FileHandler(path, mode="a")
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter(run_id=run_id))
    setattr(handler, _RUN_LOG_FLAG, True)
    root.addHandler(handler)
    return path


def reset_logging() -> None:
    """Remove the handlers we installed.  For tests."""
    root = logging.getLogger()
    for flag in (_CONSOLE_FLAG, _RUN_LOG_FLAG):
        handler = _existing(flag)
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
