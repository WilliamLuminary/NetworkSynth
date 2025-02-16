# src/utils/saver.py

import logging
import os
import pickle
import time
from typing import Any, Optional, Union

import cv2
from numpy import ndarray

from config import Config, DataType, FILE_CONFIGURATIONS, Resolution, SetName
from config.enums import FileExtension, Mode

logger = logging.getLogger(__name__)


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


def _ensure_directory(path: str, exist_ok=False) -> None:
    if not exist_ok and not os.path.exists(path):
        logger.info(f"Create new directory: {path}")
    os.makedirs(path, exist_ok=exist_ok)


class Saver:
    base_output_dir = None
    mode = None

    def __new__(cls, *args, **kwargs):
        if Config.DISABLE_SAVING:
            logger.info(f"Saving is disabled. No Saver will be instantiated. {Config.DISABLE_SAVING_NOTE}")
            return None
        return super().__new__(cls)

    def __init__(self,
                 set_name: Optional[SetName] = None,
                 resolution: Optional[Resolution] = None,
                 *,
                 output_dir: str = None):
        """
        Initialize the Saver object.
        :param output_dir:
        :param set_name: If empty, then skip this level of directory
        :param resolution: If NA, then skip this level of directory
        Preconditions:
            - Saving must be enabled (Config.DISABLE_SAVING must be False)
            - Saver.initialize() must be called before creating an instance
        Postconditions:
            - The directory for each set and resolution is created.
        """
        assert not Config.DISABLE_SAVING, "Saving is disabled."
        if output_dir:
            self.output_dir = output_dir
            self.mode = Mode.Analyze
        else:
            assert self.base_output_dir, "Base output directory is not initialized.\nCall Saver.initialize() first."
            # Each output_dir is for each original network
            self.output_dir = os.path.join(self.base_output_dir, str(set_name), str(resolution))
            self.mode = Mode.Generate

        _ensure_directory(self.output_dir, exist_ok=True)

    @classmethod
    def initialize(cls, result_dir: str = None) -> None:
        """
        Create only one base output directory for all the results.
        :return: None
        """
        if Config.DISABLE_SAVING:
            logger.info(
                f"Saving is disabled. {Config.DISABLE_SAVING_NOTE}.")
            return

        if result_dir:
            cls.mode = Mode.Analyze
            cls.base_output_dir = os.path.join(Config.BASE_OUTPUT_PATH, result_dir)
        else:
            cls.mode = Mode.Generate
            cls.base_output_dir = os.path.join(Config.BASE_OUTPUT_PATH,
                                               f'{Config.OUTPUT_DENOTE}_results_{Saver._time_id()}')
            _ensure_directory(cls.base_output_dir)
            logger.info(f"Base Output directory: {cls.base_output_dir}")
            latest_link_path = os.path.join(Config.BASE_OUTPUT_PATH, 'latest_result')
            _update_soft_link(latest_link_path, cls.base_output_dir)

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

        file_config = FILE_CONFIGURATIONS.get(data_type, FILE_CONFIGURATIONS[DataType.DEFAULT_DATA])
        file_detail = file_config.detail

        if self.mode == Mode.Generate:
            file_name_identifier = self._time_id()
            rel_path = file_config.relative_dir
            file_detail = (file_detail + '_') if file_config.detail and file_config.detail[-1] != '_' else (
                    file_config.detail or '')
        else:
            file_name_identifier = ''
            rel_path = ''

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
        with open(filepath, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)

    @staticmethod
    def _save_png_image(image: Union[ndarray], filepath: str) -> None:
        if not isinstance(image, ndarray):
            raise ValueError("Unsupported image format. Expected ndarray.")
        cv2.imwrite(filepath, image)

    @staticmethod
    def _time_id() -> str:
        return time.strftime("%Y%m%d_%H%M%S")
