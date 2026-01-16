# src/utils/saver.py

import logging
import os
from typing import Any, Optional, Union

from config import (FILE_CONFIGURATIONS, BaseConfig, DatasetId, DataType,
                    FileExtension, Mode, Resolution, SetName)

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
        # Legacy parameters - kept for backward compatibility
        set_name: Optional[SetName] = None,
        resolution: Optional[Resolution] = None,
        output_dir: str = None,
    ):
        """
        Initialize the Saver object.

        :param dataset_id: DatasetId identifying the dataset (new approach)
        :param set_name: [DEPRECATED] Use dataset_id instead
        :param resolution: [DEPRECATED] Use dataset_id instead
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

            # Handle both new DatasetId and legacy set_name/resolution
            if dataset_id is not None:
                # New approach: use DatasetId.path for directory structure
                self.output_dir = os.path.join(
                    self.base_output_dir, dataset_id.path)
            elif set_name is not None:
                # Legacy fallback
                res_str = str(resolution) if resolution else ""
                if res_str:
                    self.output_dir = os.path.join(
                        self.base_output_dir, str(set_name), res_str
                    )
                else:
                    self.output_dir = os.path.join(
                        self.base_output_dir, str(set_name))
            else:
                raise ValueError(
                    "Either dataset_id or set_name must be provided")

            self.mode = Mode.GEN

        _ensure_directory(self.output_dir, exist_ok=True)

    @classmethod
    def initialize(cls, result_dir: str = None) -> None:
        """
        Create only one base output directory for all the results.
        :return: None
        """
        if BaseConfig.DISABLE_SAVING:
            logger.info(
                f"Saving is disabled. {BaseConfig.DISABLE_SAVING_NOTE}.")
            return

        if result_dir:
            cls.mode = Mode.ANA
            cls.base_output_dir = os.path.join(
                BaseConfig.BASE_OUTPUT_PATH, result_dir)
        else:
            cls.mode = Mode.GEN
            cls.base_output_dir = os.path.join(
                BaseConfig.BASE_OUTPUT_PATH,
                f"{BaseConfig.OUTPUT_DENOTE}_results_{_time_id()}",
            )
            _ensure_directory(cls.base_output_dir)
            logger.info(f"Base Output directory: {cls.base_output_dir}")
            latest_link_path = os.path.join(
                BaseConfig.BASE_OUTPUT_PATH, "latest_result"
            )
            _update_soft_link(latest_link_path, cls.base_output_dir)

    def save_file(
        self, content: Any, data_type: DataType, file_name_prefix: Optional[str] = None
    ) -> None:
        """
        Saves the provided content to a file based on its configuration.

        :param content: The data to be saved (e.g., image or pickled object).
        :param data_type: Specifies the data type and related save configurations.
        :param file_name_prefix: Optional prefix for the generated file name.
        :return: None
        """
        if content is None:
            logger.warning("Content is None.")
            return

        if BaseConfig.DISABLE_SAVING:
            logger.warning(
                f"{BaseConfig.DISABLE_SAVING_NOTE} Saving is disabled. ")
            return

        file_config = FILE_CONFIGURATIONS.get(
            data_type, FILE_CONFIGURATIONS[DataType.DEFAULT_DATA]
        )
        file_detail = file_config.detail

        if self.mode == Mode.GEN or data_type.file_extension == FileExtension.PNG:
            file_name_identifier = _time_id()
            rel_path = file_config.relative_dir
            file_detail = (
                (file_detail + "_")
                if file_config.detail and file_config.detail[-1] != "_"
                else (file_config.detail or "")
            )
        else:
            file_name_identifier = ""
            rel_path = ""

        file_name = f"{file_name_prefix}{file_detail}{file_name_identifier}.{file_config.file_extension}"
        abs_path = os.path.join(self.output_dir, rel_path, file_name)
        _ensure_directory(os.path.dirname(abs_path), exist_ok=True)

        if FileExtension.PNG == data_type.file_extension:
            Saver._save_png_image(content, abs_path)
        elif FileExtension.PKL == data_type.file_extension:
            Saver._save_pickle(content, abs_path)
        else:
            raise ValueError(f"Unsupported DataType for saving: {data_type}")

        logger.info(f"Saved file: {abs_path}")

    @staticmethod
    def _save_pickle(obj: Any, filepath: str) -> None:
        with open(filepath, "wb") as f:
            import pickle

            # noinspection PyTypeChecker
            pickle.dump(obj, f)

    @staticmethod
    def _save_png_image(image, filepath: str) -> None:
        from numpy import ndarray

        assert isinstance(
            image, ndarray), "Unsupported image format. Expected ndarray."

        import cv2

        cv2.imwrite(filepath, image)


def _update_soft_link(link_path: str, target_path: str) -> None:
    if os.path.exists(link_path) or os.path.islink(link_path):
        try:
            os.unlink(link_path)
        except OSError as e:
            logger.warning(
                f"Failed to remove existing soft link: {link_path}. Error: {e}"
            )
            return
    try:
        os.symlink(target_path, link_path)
    except OSError as e:
        # On Windows, symlinks require admin privileges or Developer Mode
        logger.warning(f"Failed to create soft link: {link_path}. Error: {e}")
        return


def _ensure_directory(path: str, exist_ok=False) -> None:
    if not exist_ok and not os.path.exists(path):
        logger.info(f"Create new directory: {path}")
    os.makedirs(path, exist_ok=exist_ok)


def _time_id() -> str:
    import time

    return time.strftime("%Y%m%d_%H%M%S")
