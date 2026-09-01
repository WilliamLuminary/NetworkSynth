# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import logging
import os
from typing import Optional

_CONSOLE_FLAG = "_networksynth_console"
_RUN_LOG_FLAG = "_networksynth_run_log"

_CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _existing(flag: str) -> Optional[logging.Handler]:
    for handler in logging.getLogger().handlers:
        if getattr(handler, flag, False):
            return handler
    return None


def configure_console(level: int = logging.INFO) -> None:
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
    from utils import JsonFormatter

    root = logging.getLogger()
    root.setLevel(level)

    previous = _existing(_RUN_LOG_FLAG)
    if previous is not None:
        root.removeHandler(previous)
        previous.close()

    if not os.path.isdir(run_root):
        return None

    path = os.path.join(run_root, "run.jsonl")
    handler = logging.FileHandler(path, mode="a")
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter(run_id=run_id))
    setattr(handler, _RUN_LOG_FLAG, True)
    root.addHandler(handler)
    return path


def reset_logging() -> None:
    root = logging.getLogger()
    for flag in (_CONSOLE_FLAG, _RUN_LOG_FLAG):
        handler = _existing(flag)
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
