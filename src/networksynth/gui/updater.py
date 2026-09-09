# SPDX-License-Identifier: GPL-3.0-or-later
"""What version this is, and how to move it forward.

A checkout on a normal branch pulls. A `dist` checkout, which is what another
project carries as a submodule, has no upstream to pull from and is moved by
fetching the branch and resetting onto it.
"""

import os
import subprocess
from typing import List, Tuple

from networksynth import __version__

_TIMEOUT = 30


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
            stdin=subprocess.DEVNULL,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""},
        )
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


def in_checkout() -> bool:
    return _git("rev-parse", "--is-inside-work-tree")[0] == 0


def version() -> str:
    """The tag on this exact commit, or the number the package was published with.

    A shallow submodule fetches a commit, not the tags that name it, so git has
    nothing to answer with there. publish_dist.sh stamps __version__ from the tag
    it builds, which is why the fallback is a fact rather than a guess.

    Anything between tags in a working clone is an untagged state, and git says
    so; inventing a number for it would claim more than is known.
    """
    code, tag = _git("describe", "--tags", "--exact-match")
    if code == 0:
        return tag
    if _has_tags():
        return ""
    return __version__


def _has_tags() -> bool:
    """Whether git has any tag to describe with; a shallow fetch brings none.

    _git folds stderr into its output, so the exit code is what says whether
    there was a repository to ask at all.
    """
    code, listed = _git("tag", "-l")
    return code == 0 and bool(listed)


def _branch() -> str:
    code, name = _git("rev-parse", "--abbrev-ref", "HEAD")
    return name if code == 0 else ""


def channel() -> str:
    """The branch this checkout follows.

    A submodule is checked out at a commit rather than a branch, so when there is
    no branch name the tag says which line of releases this came from.
    """
    branch = _branch()
    if branch and branch != "HEAD":
        return branch
    tag = version()
    if not tag:
        return ""
    return "dist-dev" if tag.endswith("-dev") else "dist"


def update_commands() -> List[List[str]]:
    """Always the newest of this checkout's own channel: dist fetches dist,
    dist-dev fetches dist-dev, and a working branch pulls its own upstream."""
    name = channel()
    if not name:
        return []
    if name.startswith("dist"):
        # --tags as well, or a shallow submodule clone has no tag to read the
        # version from and the footer stays blank.
        return [
            ["fetch", "origin", name, "--depth", "1", "--tags"],
            ["checkout", "-B", name, "FETCH_HEAD"],
        ]
    return [["pull", "--ff-only"]]


def update() -> Tuple[bool, str, bool]:
    """Move the checkout forward. Returns whether it worked and what to show."""
    if not in_checkout():
        return False, "Not a git checkout, so there is nothing to update.", False

    commands = update_commands()
    if not commands:
        return (
            False,
            "This checkout is not on a branch and carries no tag, so there is "
            "nothing to follow.",
            False,
        )

    before = _git("rev-parse", "--short", "HEAD")[1]
    for command in commands:
        code, output = _git(*command)
        if code != 0:
            return False, (output.splitlines()[-1] if output else "git failed"), False

    after = _git("rev-parse", "--short", "HEAD")[1]
    if before == after:
        return True, f"Already the latest version ({version() or after}).", False
    return (
        True,
        f"Updated to {version() or after}. Restart NetworkSynth to load it.",
        True,
    )
