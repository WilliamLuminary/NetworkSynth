# SPDX-License-Identifier: GPL-3.0-or-later
"""Console-script entry points declared in pyproject.toml."""

import sys


def main_gui() -> int:
    from networksynth.gui.app import main

    return main()


def main_cli() -> int:
    from networksynth.run import main

    return main(sys.argv)


def main_run_spec() -> int:
    from networksynth.gui_run import main

    return main(sys.argv)


def main_analyse() -> int:
    from networksynth.analyse import main

    return main(sys.argv)
