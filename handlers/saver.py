# src/handlers/saver.py

import logging
import os
import sys
from typing import Any, Optional

from configs import (
    BaseConfig,
    DatasetId,
    Mode,
)
from configs.base_config import tagged

logger = logging.getLogger(__name__)


class Saver:
    base_output_dir = None
    mode = None

    def __new__(cls, *args, **kwargs):
        if BaseConfig.DISABLE_SAVING:
            logger.info(
                f"Saving is disabled. No Saver will be instantiated. "
                f"{BaseConfig.DISABLE_SAVING_NOTE}"
            )
            return None
        return super().__new__(cls)

    def __init__(
        self,
        dataset_id: Optional[DatasetId] = None,
        *,
        output_dir: str = None,
    ):
        """
        Initialize the Saver object.

        :param dataset_id: DatasetId identifying the dataset
        :param output_dir: Direct output directory path (for analysis mode)

        Preconditions:
            - Saving must be enabled (Config.DISABLE_SAVING must be False)
            - Saver.initialize() must be called before creating an instance

        Postconditions:
            - The directory for the dataset is created.
        """
        assert not BaseConfig.DISABLE_SAVING, "Saving is disabled."
        if output_dir:
            assert (
                not self.base_output_dir
            ), "Output directory has been set. Don't call Saver.initialize()."
            self.output_dir = output_dir
            self.mode = Mode.ANA
        else:
            assert (
                self.base_output_dir
            ), "Base output directory is not initialized.\nCall Saver.initialize() first."

            if dataset_id is None:
                raise ValueError("dataset_id must be provided")

            self.output_dir = os.path.join(self.base_output_dir, dataset_id.path)
            self.mode = Mode.GEN

        _ensure_directory(self.output_dir, exist_ok=True)
        self._save_func = BaseConfig.save

    @classmethod
    def initialize(cls, result_dir: str = None) -> None:
        """
        Create only one base output directory for all the results.
        :return: None
        """
        if BaseConfig.DISABLE_SAVING:
            logger.info(f"Saving is disabled. {BaseConfig.DISABLE_SAVING_NOTE}.")
            return

        if result_dir:
            cls.mode = Mode.ANA
            cls.base_output_dir = os.path.join(BaseConfig.BASE_OUTPUT_PATH, result_dir)
        else:
            cls.mode = Mode.GEN
            cls.base_output_dir = os.path.join(
                BaseConfig.BASE_OUTPUT_PATH,
                f"{BaseConfig.OUTPUT_DENOTE}_results_{_time_id()}",
            )
            _ensure_directory(cls.base_output_dir)
            logger.info(
                f"Base Output directory: {cls.base_output_dir}", extra=tagged("IO")
            )
            latest_link_path = os.path.join(
                BaseConfig.BASE_OUTPUT_PATH, "latest_result"
            )
            _update_soft_link(latest_link_path, cls.base_output_dir)

    _batch_timestamp: Optional[str] = None

    @classmethod
    def begin_batch(cls) -> str:
        """Set a shared timestamp for a group of related saves.

        All ``save`` calls until ``end_batch`` will share this
        timestamp.  Returns the generated timestamp so callers can
        log it.
        """
        cls._batch_timestamp = _time_id()
        return cls._batch_timestamp

    @classmethod
    def end_batch(cls) -> None:
        """Clear the batch timestamp."""
        cls._batch_timestamp = None

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

        if BaseConfig.DISABLE_SAVING:
            logger.warning(f"{BaseConfig.DISABLE_SAVING_NOTE} Saving is disabled.")
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
