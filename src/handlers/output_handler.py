# src/utils/output_handler.py

import datetime
import logging
import os
import pickle
from typing import Any, Union

import cv2
from numpy import ndarray

from config.base import BaseConfig
from config.enums import DataType, FileTag
from config.name_resolution_set import NameResolutionSet
from utils.debug_utils import time_id


class OutputHandler(BaseConfig):
    base_output_dir = os.path.join(BaseConfig.BASE_OUTPUT_PATH, f'results_{time_id()}')

    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.name_res_output_dir = os.path.join(self.base_output_dir, str(name_res_set.set_name),
                                                str(name_res_set.resolution))
        self.ensure_directory(self.name_res_output_dir)

    @classmethod
    def initialize(cls):
        # cls.archive_if_exists(cls.OUTPUT_DIR)
        cls.ensure_directory(cls.base_output_dir)
        print(f"Created new directory: {cls.base_output_dir}")

    @staticmethod
    def archive_if_exists(path: str) -> None:
        if os.path.exists(path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.isfile(path):
                base, ext = os.path.splitext(path)
                archived_path = f"{base}_archived_{timestamp}{ext}"
                os.rename(path, archived_path)
                logging.info(f"Archived existing file: {archived_path}")
            elif os.path.isdir(path):
                archived_path = f"{path}_archived_{timestamp}"
                os.rename(path, archived_path)
                logging.info(f"Archived existing directory: {archived_path}")
        else:
            logging.warning(f"Path does not exist, nothing to archive: {path}")

    @staticmethod
    def ensure_directory(path: str, exist_ok=False) -> None:
        os.makedirs(path, exist_ok=exist_ok)
        logging.info(f"Ensured directory exists: {path}")

    @staticmethod
    def delete_file(filepath: str) -> None:
        if os.path.exists(filepath) and os.path.isfile(filepath):
            os.remove(filepath)
            logging.info(f"Deleted existing file: {filepath}")
        elif os.path.isdir(filepath):
            logging.warning(f"Expected a file but found a directory at: {filepath}")
        else:
            logging.debug(f"No existing file to delete at: {filepath}")

    def save_file(self, content: Any, data_type: DataType) -> None:
        file_config = BaseConfig.FILE_CONFIGURATIONS[data_type]
        rel_path = file_config.relative_dir
        file_name = f"{time_id()}.{file_config.file_type}"
        abs_path = os.path.join(self.name_res_output_dir, rel_path, file_name)
        self.ensure_directory(os.path.dirname(abs_path))

        if FileTag.FIG in data_type.tags:
            OutputHandler._save_image(content, abs_path)
        elif FileTag.PKL in data_type.tags:
            OutputHandler._save_pickle(content, abs_path)
        else:
            raise ValueError(f"Unsupported DataType for saving: {data_type}")

    @staticmethod
    def _save_pickle(obj: Any, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)
        logging.info(f"Saved pickle file: {filepath}")

    @staticmethod
    def _save_image(image: Union[ndarray], filepath: str) -> None:
        if not isinstance(image, ndarray):
            raise ValueError("Unsupported image format. Expected ndarray or plt.Figure.")

        cv2.imwrite(filepath, image)
        logging.info(f"Saved image: {filepath}")
