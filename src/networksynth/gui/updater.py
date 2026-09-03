# SPDX-License-Identifier: GPL-3.0-or-later
"""What version this is, and how to move it forward.

A checkout on a normal branch pulls. A `dist` checkout, which is what another
project carries as a submodule, has no upstream to pull from and is moved by
fetching the branch and resetting onto it.
"""

import os
import subprocess
from typing import List, Tuple

_TIMEOUT = 120


def repo_root() -> str:
    """The checkout holding this package, whether that is the repository or a submodule."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _git(*args: str) -> Tuple[int, str]:
    try:
        done = subprocess.run(
            ["git", "-C", repo_root(), *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


def in_checkout() -> bool:
    return _git("rev-parse", "--is-inside-work-tree")[0] == 0


def version() -> str:
    """The tag on this exact commit, or nothing.

    Anything between tags is an untagged state, and inventing a number for it
    would say more than is known.
    """
    code, tag = _git("describe", "--tags", "--exact-match")
    return tag if code == 0 else ""


def _branch() -> str:
    code, name = _git("rev-parse", "--abbrev-ref", "HEAD")
    return name if code == 0 else ""


def update_commands() -> List[List[str]]:
    """A dist checkout is moved by fetching its branch; anything else pulls."""
    branch = _branch()
    if branch.startswith("dist"):
        return [
            ["fetch", "origin", branch, "--depth", "1"],
            ["checkout", "-B", branch, "FETCH_HEAD"],
        ]
    return [["pull", "--ff-only"]]


def update() -> Tuple[bool, str]:
    """Move the checkout forward. Returns whether it worked and what to show."""
    if not in_checkout():
        return False, "Not a git checkout, so there is nothing to update."

    before = _git("rev-parse", "--short", "HEAD")[1]
    for command in update_commands():
        code, output = _git(*command)
        if code != 0:
            return False, output.splitlines()[-1] if output else "git failed"

    after = _git("rev-parse", "--short", "HEAD")[1]
    if before == after:
        return True, f"Already up to date ({version()})."
    return True, f"Updated to {version()}. Restart to load it."
