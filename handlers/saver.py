import logging
import os
import sys
from typing import Any, Optional

from utils import tagged

logger = logging.getLogger(__name__)


class Saver:
    """Writes one dataset's outputs into one directory.

    Where the run writes is passed in as a value, so nothing here is global and
    many runs can coexist.  A Saver always writes: whether a run saves at all is
    decided once by :func:`build_saver`, which returns a :class:`NullSaver`
    instead when ``DISABLE_SAVING`` is set.
    """

    def __init__(self, config, out_dir: str):
        """
        :param config: The active config, supplying the save specs
        :param out_dir: The directory this saver writes into

        Postconditions:
            - ``out_dir`` exists.
        """
        self.output_dir = out_dir
        _ensure_directory(self.output_dir, exist_ok=True)
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
        self._batch_timestamp = None

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        """Save *content* according to the specs returned by the config.

        :param content: The data to be saved.
        :param identifier: String identifier (e.g. ``"original_network"``).
            Resolves to ``SAVE_<IDENTIFIER>`` on the active config.
        :param prefix: Optional prefix for the generated file name.
        """
        if content is None:
            logger.warning("Content is None.")
            return

        for spec in self._save_func(identifier):
            if spec.use_timestamp:
                file_id = self._batch_timestamp or _time_id()
                detail_part = f"{prefix}{spec.detail}_" if spec.detail else prefix
            else:
                file_id = ""
                detail_part = f"{prefix}{spec.detail}" if spec.detail else prefix

            file_name = f"{detail_part}{file_id}.{spec.extension}"
            abs_path = os.path.join(self.output_dir, spec.relative_dir, file_name)
            _ensure_directory(os.path.dirname(abs_path), exist_ok=True)

            spec.save_fn(content, abs_path)
            logger.info(f"Saved file: {abs_path}", extra=tagged("IO"))


class NullSaver(Saver):
    """A Saver that accepts everything and writes nothing.

    ``DISABLE_SAVING`` used to yield ``None``, which every caller had to
    remember to check.  Keeps ``output_dir`` only, because callers build
    subdirectory paths from it; nothing is created on disk.
    """

    def __init__(self, out_dir: str):
        self.output_dir = out_dir

    def begin_batch(self) -> str:
        return ""

    def end_batch(self) -> None:
        pass

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        logger.debug(f"Saving disabled — skipped {prefix}{identifier}")


def build_saver(config, out_dir: str) -> Saver:
    """The saver this run should use: a real one, or a no-op.

    The single place ``DISABLE_SAVING`` decides anything about saving.  The
    other place it is read is :func:`~handlers.run_paths.create_run_paths`,
    which decides whether the directory is created at all.
    """
    if config.DISABLE_SAVING:
        logger.info(
            f"Saving is disabled; nothing will be written. {config.DISABLE_SAVING_NOTE}"
        )
        return NullSaver(out_dir)
    return Saver(config, out_dir)


def _is_junction(path: str) -> bool:
    try:
        import stat

        return bool(
            os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    except (AttributeError, OSError):
        return False


_IS_WINDOWS = sys.platform == "win32"


def _create_junction(link_path: str, target_path: str) -> None:
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
