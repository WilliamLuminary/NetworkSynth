# src/utils/saver.py

import datetime
import logging
import os
import pickle
import time
from typing import Any, Optional, Union

import cv2
from numpy import ndarray

from config import Config, DataType, FileTag, NameResolutionSet

logger = logging.getLogger(__name__)


class Saver:
    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.name_res_output_dir = os.path.join(self.base_output_dir, str(name_res_set.set_name),
                                                str(name_res_set.resolution))
        self.ensure_directory(self.name_res_output_dir, exist_ok=True)

    @classmethod
    def initialize(cls):
        cls.base_output_dir = os.path.join(Config.BASE_OUTPUT_PATH, f'results_{Saver._time_id()}')
        cls.ensure_directory(cls.base_output_dir)
        logger.info(f"Created new directory: {cls.base_output_dir}")

        cls.latest_link_path = os.path.join(Config.BASE_OUTPUT_PATH, 'latest_result')
        Saver._update_soft_link(cls.latest_link_path, cls.base_output_dir)

    @staticmethod
    def _update_soft_link(link_path: str, target_path: str) -> None:
        if os.path.exists(link_path) or os.path.islink(link_path):
            try:
                os.unlink(link_path)
                logger.info(f"Removed existing soft link: {link_path}")
            except OSError as e:
                logger.error(f"Failed to remove existing soft link: {link_path}. Error: {e}")
                raise
        try:
            os.symlink(target_path, link_path)
            logger.info(f"Created new soft link: {link_path} -> {target_path}")
        except OSError as e:
            logger.error(f"Failed to create soft link: {link_path}. Error: {e}")
            raise

    @staticmethod
    def archive_if_exists(path: str) -> None:
        if os.path.exists(path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.isfile(path):
                base, ext = os.path.splitext(path)
                archived_path = f"{base}_archived_{timestamp}{ext}"
                os.rename(path, archived_path)
                logger.info(f"Archived existing file: {archived_path}")
            elif os.path.isdir(path):
                archived_path = f"{path}_archived_{timestamp}"
                os.rename(path, archived_path)
                logger.info(f"Archived existing directory: {archived_path}")
        else:
            logger.warning(f"Path does not exist, nothing to archive: {path}")

    @staticmethod
    def ensure_directory(path: str, exist_ok=False) -> None:
        flag = False
        if not exist_ok and not os.path.exists(path):
            flag = True
        os.makedirs(path, exist_ok=exist_ok)
        if flag:
            logger.info(f"Created new directory: {path}")

    @staticmethod
    def delete_file(filepath: str) -> None:
        if os.path.exists(filepath) and os.path.isfile(filepath):
            os.remove(filepath)
            logger.info(f"Deleted existing file: {filepath}")
        elif os.path.isdir(filepath):
            logger.warning(f"Expected a file but found a directory at: {filepath}")
        else:
            logger.debug(f"No existing file to delete at: {filepath}")

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

        file_config = Config.FILE_CONFIGURATIONS[data_type]
        rel_path = file_config.relative_dir

        file_name_prefix = (file_name_prefix + '_') if file_name_prefix and file_name_prefix[-1] != '_' else (
                file_name_prefix or '')
        file_detail = (file_config.detail + '_') if file_config.detail and file_config.detail[-1] != '_' else (
                file_config.detail or '')
        file_name = f"{file_name_prefix}{file_detail}{self._time_id()}.{file_config.file_extension}"

        abs_path = os.path.join(self.name_res_output_dir, rel_path, file_name)
        self.ensure_directory(os.path.dirname(abs_path), exist_ok=True)

        if FileTag.FIG in data_type.tags:
            Saver._save_image(content, abs_path)
        elif FileTag.DATA in data_type.tags:
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
