# SPDX-License-Identifier: GPL-3.0-or-later
from enum import Enum, auto


class AnalysisMode(Enum):
    BASIC = auto()
    FULL = auto()
    RETROACTIVE = auto()
