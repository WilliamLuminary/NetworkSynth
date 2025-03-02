import os
from typing import Optional


def find_file_with_pattern(directory_path, pattern, details='', must_exist=True) -> Optional[str]:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files_in_directory = os.listdir(directory_path)
    matched_files = [file_name for file_name in files_in_directory if pattern.search(file_name)]

    if len(matched_files) == 1:
        file_path = os.path.join(directory_path, matched_files[0])
        return file_path
    elif len(matched_files) > 1:
        raise FileExistsError(f"Multiple {details} files found: {matched_files}.")
    else:
        assert not must_exist, f"No {details} file found in {directory_path}."
        return None
