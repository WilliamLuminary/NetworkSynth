# src/utils/output_handler.py

import datetime
import logging
import os
import pickle
import time
from typing import Any, Union

import cv2
from jupyter_server.serverapp import flags
from numpy import ndarray

from config.config import Config
from config.enums import DataType, FileTag
from config.name_resolution_set import NameResolutionSet
from config import Config, DataType, FileTag, NameResolutionSet


class OutputHandler(Config):

    def __init__(self, name_res_set: NameResolutionSet):
        self.name_res_set = name_res_set
        self.name_res_output_dir = os.path.join(self.base_output_dir, str(name_res_set.set_name),
                                                str(name_res_set.resolution))
        self.ensure_directory(self.name_res_output_dir, exist_ok=True)

    @classmethod
    def initialize(cls):
        cls.base_output_dir = os.path.join(Config.BASE_OUTPUT_PATH, f'results_{OutputHandler._time_id()}')
        cls.ensure_directory(cls.base_output_dir)

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
        file_config = Config.FILE_CONFIGURATIONS[data_type]
        rel_path = file_config.relative_dir
        file_name = f"{data_type}_{self._time_id()}.{data_type.file_extension}"
        abs_path = os.path.join(self.name_res_output_dir, rel_path, file_name)
        self.ensure_directory(os.path.dirname(abs_path), exist_ok=True)

        if FileTag.FIG in data_type.tags:
            OutputHandler._save_image(content, abs_path)
        elif FileTag.DATA in data_type.tags:
            OutputHandler._save_pickle(content, abs_path)
        else:
            raise ValueError(f"Unsupported DataType for saving: {data_type}")

    @staticmethod
    def _save_pickle(obj: Any, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)

    @staticmethod
    def _save_image(image: ndarray, filepath: str) -> None:
        cv2.imwrite(filepath, image)

    @staticmethod
    def _time_id() -> str:
        return time.strftime("%Y%m%d_%H%M%S")
