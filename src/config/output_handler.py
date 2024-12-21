# src/utils/output_handler.py

import datetime
import logging
import os
import pickle
from typing import Any

from config.base import BaseConfig
from config.name_resolution_set import NameResolutionSet


class OutputHandler(BaseConfig):
    def __init__(self, image_set: NameResolutionSet):
        self.base_output_dir = self.OUTPUT_DIR
        self.synthetic_graph_path = os.path.join(self.OUTPUT_DIR, image_set.set_name.value, self.SYNTHETIC_GRAPH_DIRECTORY_NAME)
        self.original_graph_path = os.path.join(self.OUTPUT_DIR, image_set.resolution.value, self.ORIGINAL_GRAPH_DIRECTORY_NAME)

        self.archive_if_exists(self.base_output_dir)
        self.ensure_directory(self.base_output_dir)
        print(f"Created new directory: {self.base_output_dir}")

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
    def ensure_directory(path: str) -> None:
        os.makedirs(path, exist_ok=True)
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

    @staticmethod
    def save_file(content: bytes, filepath: str, archive_existing: bool = False) -> None:
        OutputHandler.ensure_directory(os.path.dirname(filepath))
        if archive_existing:
            OutputHandler.archive_if_exists(filepath)
        else:
            OutputHandler.delete_file(filepath)
        with open(filepath, 'wb') as f:
            f.write(content)
        logging.info(f"Saved file: {filepath}")

    @staticmethod
    def save_pickle(obj: Any, filepath: str, archive_existing: bool = False) -> None:
        OutputHandler.ensure_directory(os.path.dirname(filepath))
        if archive_existing:
            OutputHandler.archive_if_exists(filepath)
        else:
            OutputHandler.delete_file(filepath)
        with open(filepath, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(obj, f)
        logging.info(f"Saved pickle file: {filepath}")
