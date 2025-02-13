# src/utils/saver.py

import logging
import os
import pickle
import time
from typing import Any, Optional, Union

import cv2
from numpy import ndarray

from config import Config, DataType, FILE_CONFIGURATIONS, FileTag, Resolution, SetName

logger = logging.getLogger(__name__)


class Saver:
    base_output_dir = None
    latest_link_path = None

    def __new__(cls, *args, **kwargs):
        if Config.DISABLE_SAVING:
            logger.info(f"Saving is disabled. No Saver will be instantiated. {Config.DISABLE_SAVING_NOTE}")
            return None
        return super().__new__(cls)

    def __init__(self, set_name, resolution):
        """
        Initialize the Saver object.
        The directory for each set and resolution is created.
        :param set_name: If empty, then skip this level of directory
        :param resolution: If NA, then skip this level of directory
        Preconditions:
            - Saving must be enabled (Config.DISABLE_SAVING must be False)
            - Saver.initialize() must be called before creating an instance
        """
        assert not Config.DISABLE_SAVING, "Saving is disabled."
        assert self.base_output_dir and self.latest_link_path, \
            "Base output directory is not initialized.\nCall Saver.initialize() first."

        self.name_res_output_dir = os.path.join(self.base_output_dir, str(set_name), str(resolution))
        self.ensure_directory(self.name_res_output_dir, exist_ok=True)

    @classmethod
    def initialize(cls) -> None:
        """
        Create only one base output directory for all the results.
        :return: None
        """
        if Config.DISABLE_SAVING:
            logger.info(
                f"Saving is disabled. {Config.DISABLE_SAVING_NOTE}. The base output directory will not be created.")
            return

        cls.base_output_dir = os.path.join(Config.BASE_OUTPUT_PATH, f'{Config.OUTPUT_DENOTE}_results_{Saver._time_id()}')
        cls.ensure_directory(cls.base_output_dir)
        logger.info(f"Created Base Output directory: {cls.base_output_dir}")

        cls.latest_link_path = os.path.join(Config.BASE_OUTPUT_PATH, 'latest_result')
        Saver._update_soft_link(cls.latest_link_path, cls.base_output_dir)

    @staticmethod
    def _update_soft_link(link_path: str, target_path: str) -> None:
        if os.path.exists(link_path) or os.path.islink(link_path):
            try:
                os.unlink(link_path)
            except OSError as e:
                logger.error(f"Failed to remove existing soft link: {link_path}. Error: {e}")
                raise
        try:
            os.symlink(target_path, link_path)
        except OSError as e:
            logger.error(f"Failed to create soft link: {link_path}. Error: {e}")
            raise

    @staticmethod
    def ensure_directory(path: str, exist_ok=False) -> None:
        flag = False
        if not exist_ok and not os.path.exists(path):
            flag = True
        os.makedirs(path, exist_ok=exist_ok)
        if flag:
            logger.info(f"Created new directory: {path}")

    def save_file(self, content: Any, data_type: DataType, file_name_prefix: Optional[str] = None) -> None:
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

        if Config.DISABLE_SAVING:
            logger.warning(f"{Config.DISABLE_SAVING_NOTE} Saving is disabled. ")
            return

        file_config = FILE_CONFIGURATIONS[data_type]
        rel_path = file_config.relative_dir
        file_detail = (file_config.detail + '_') if file_config.detail and file_config.detail[-1] != '_' else (
                file_config.detail or '')
        file_name = f"{file_name_prefix}{file_detail}{self._time_id()}.{file_config.file_extension}"

        abs_path = os.path.join(self.name_res_output_dir, rel_path, file_name)
        self.ensure_directory(os.path.dirname(abs_path), exist_ok=True)

        if FileTag.FIG in data_type.tags:
            Saver._save_image(content, abs_path)
        elif FileTag.DATA in data_type.tags:
            if hasattr(content, 'as_savable') and callable(content.as_savable):
                content = content.as_savable()
            Saver._save_pickle(content, abs_path)
        else:
            raise ValueError(f"Unsupported DataType for saving: {data_type}")

        logger.info(f"Saved file: {abs_path}")

    @staticmethod
    def _save_pickle(obj: Any, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)

    @staticmethod
    def _save_image(image: Union[ndarray], filepath: str) -> None:
        if not isinstance(image, ndarray):
            raise ValueError("Unsupported image format. Expected ndarray.")

        cv2.imwrite(filepath, image)

    @staticmethod
    def _time_id() -> str:
        return time.strftime("%Y%m%d_%H%M%S")
