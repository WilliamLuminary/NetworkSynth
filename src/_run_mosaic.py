"""Temporary launcher for mosaic pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import main_mosaic  # noqa: E402

main_mosaic.main()
