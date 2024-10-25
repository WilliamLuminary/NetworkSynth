# src/utils/output_handler.py

import datetime
import os


class OutputHandler:
    def __init__(self):
        pass

    @staticmethod
    def archive_if_exists(path):
        """
        Archive the existing file or directory at the given path by renaming it with a timestamp.
        """
        if os.path.exists(path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.isfile(path):
                base, ext = os.path.splitext(path)
                archived_path = f"{base}_archived_{timestamp}{ext}"
                os.rename(path, archived_path)
                print(f"Archived existing file: {archived_path}")
            elif os.path.isdir(path):
                archived_path = f"{path}_archived_{timestamp}"
                os.rename(path, archived_path)
                print(f"Archived existing directory: {archived_path}")

    @staticmethod
    def ensure_directory(path):
        """
        Ensure that the directory exists.
        """
        os.makedirs(path, exist_ok=True)

    def save_file(self, content, filepath, mode='wb'):
        """
        Save content to a file, archiving any existing file first.
        """
        self.ensure_directory(os.path.dirname(filepath))
        self.archive_if_exists(filepath)
        with open(filepath, mode) as f:
            f.write(content)
        print(f"Saved file: {filepath}")

    def save_pickle(self, obj, filepath):
        """
        Save an object to a pickle file, archiving any existing file first.
        """
        import pickle
        self.ensure_directory(os.path.dirname(filepath))
        self.archive_if_exists(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved pickle file: {filepath}")
