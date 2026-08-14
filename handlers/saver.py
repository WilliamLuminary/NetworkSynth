# src/handlers/saver.py

import logging
import os
import sys
from typing import Any, Optional

from configs.base_config import tagged

logger = logging.getLogger(__name__)


class Saver:
    """Writes one dataset's outputs into one directory.

    Holds only ``(config, out_dir)``.  Where the run writes is decided by
    :func:`~handlers.run_paths.create_run_paths` and passed in as a value, so
    nothing here is global and many runs can coexist.

    A Saver that exists always writes.  Whether a run saves at all is decided
    once, by :class:`~handlers.run_agent.RunAgent`, which simply does not build
    one when ``DISABLE_SAVING`` is set.
    """

    def __init__(self, config, out_dir: str):
        """
        :param config: The active config, supplying the save specs
        :param out_dir: The directory this saver writes into

        Postconditions:
            - ``out_dir`` exists.
        """
        self._config = config
        self.output_dir = out_dir
        _ensure_directory(self.output_dir, exist_ok=True)
        # Bound to the real config class, so `save_<identifier>` overrides
        # resolve through the normal MRO with no injection required.
        self._save_func = config.save
        self._batch_timestamp: Optional[str] = None

    def begin_batch(self) -> str:
        """Set a shared timestamp for a group of related saves.

        All ``save`` calls until ``end_batch`` will share this
        timestamp.  Returns the generated timestamp so callers can
        log it.
        """
        self._batch_timestamp = _time_id()
        return self._batch_timestamp

    def end_batch(self) -> None:
        """Clear the batch timestamp."""
        self._batch_timestamp = None

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        """Save *content* according to the specs returned by the config.

        :param content: The data to be saved.
        :param identifier: String identifier (e.g. ``"original_network"``).
            Maps to ``save_<identifier>`` on the active config.
        :param prefix: Optional prefix for the generated file name.
        """
        if content is None:
            logger.warning("Content is None.")
            return

        specs = self._save_func(identifier)
        for spec in specs:
            relative_dir, detail, extension, save_fn = spec[:4]
            use_timestamp = spec[4] if len(spec) > 4 else True

            if use_timestamp:
                file_id = self._batch_timestamp or _time_id()
                detail_part = f"{prefix}{detail}_" if detail else prefix
            else:
                file_id = ""
                detail_part = f"{prefix}{detail}" if detail else prefix

            file_name = f"{detail_part}{file_id}.{extension}"
            abs_path = os.path.join(self.output_dir, relative_dir, file_name)
            _ensure_directory(os.path.dirname(abs_path), exist_ok=True)

            save_fn(content, abs_path)
            logger.info(f"Saved file: {abs_path}", extra=tagged("IO"))


def _is_junction(path: str) -> bool:
    """Check whether *path* is an NTFS directory junction (reparse point)."""
    try:
        import stat

        return bool(
            os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    except (AttributeError, OSError):
        return False


_IS_WINDOWS = sys.platform == "win32"


def _create_junction(link_path: str, target_path: str) -> None:
    """Create an NTFS directory junction (no elevated privileges needed)."""
    try:
        import _winapi  # CPython C extension, available on all Windows builds

        _winapi.CreateJunction(target_path, link_path)
        return
    except ImportError:
        pass

    import subprocess

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", link_path, target_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _update_soft_link(link_path: str, target_path: str) -> None:
    """Point *link_path* at *target_path*.

    Unix: symbolic link.  Windows: directory junction (avoids the need
    for elevated privileges or Developer Mode).
    """
    if os.path.islink(link_path):
        try:
            os.unlink(link_path)
        except OSError as e:
            logger.warning(f"Failed to remove existing link: {link_path}. Error: {e}")
            return
    elif _IS_WINDOWS and _is_junction(link_path):
        try:
            os.rmdir(link_path)
        except OSError as e:
            logger.warning(
                f"Failed to remove existing junction: {link_path}. Error: {e}"
            )
            return

    try:
        if _IS_WINDOWS:
            _create_junction(link_path, os.path.abspath(target_path))
        else:
            rel_target = os.path.relpath(target_path, os.path.dirname(link_path))
            os.symlink(rel_target, link_path)
    except Exception as e:
        logger.warning(
            f"Failed to create link: {link_path} -> {target_path}. Error: {e}"
        )


def _ensure_directory(path: str, exist_ok=False) -> None:
    if not exist_ok and not os.path.exists(path):
        logger.info(f"Create new directory: {path}", extra=tagged("IO"))
    os.makedirs(path, exist_ok=exist_ok)


def _time_id() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")
