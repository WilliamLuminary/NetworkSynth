# SPDX-License-Identifier: GPL-3.0-or-later
import os
import re
from typing import Optional


def find_file_with_pattern(
    directory_path, pattern, details="", must_exist=True
) -> Optional[str]:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Directory not found: {directory_path}. Regex pattern: {pattern}"
        )

    regex_pattern = re.compile(pattern, re.IGNORECASE)
    matched_files = [
        file_name
        for file_name in os.listdir(directory_path)
        if regex_pattern.search(file_name)
    ]

    if len(matched_files) == 1:
        return os.path.join(directory_path, matched_files[0])
    if len(matched_files) > 1:
        raise FileExistsError(
            f"Multiple {details} files found: {matched_files}. Regex pattern: {pattern}"
        )
    assert (
        not must_exist
    ), f"No {details} file found in {directory_path}. Regex pattern: {pattern}"
    return None
