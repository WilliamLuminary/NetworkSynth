import logging
import os
import sys
from typing import Any, Optional

from utils import tagged

logger = logging.getLogger(__name__)


class Saver:

    def __init__(self, config, out_dir: str):
        self.output_dir = out_dir
        _ensure_directory(self.output_dir, exist_ok=True)
        self._save_func = config.save
        self._batch_timestamp: Optional[str] = None

    def begin_batch(self) -> str:
        self._batch_timestamp = _time_id()
        return self._batch_timestamp

    def end_batch(self) -> None:
        self._batch_timestamp = None

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
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

    def __init__(self, out_dir: str):
        self.output_dir = out_dir

    def begin_batch(self) -> str:
        return ""

    def end_batch(self) -> None:
        pass

    def save(self, content: Any, identifier: str, prefix: str = "") -> None:
        logger.debug(f"Saving disabled — skipped {prefix}{identifier}")


def build_saver(config, out_dir: str) -> Saver:
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
        import _winapi

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
