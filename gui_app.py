# gui_app.py
"""NetworkSynth's own GUI entry point.

    python gui_app.py

Opens the synthesis window.  It writes a run-spec and launches ``gui_run.py``
as a subprocess, so both entry points share one code path — and when
StructuralGT's button opens this window later, nothing here changes.
"""

import sys

from gui.app import main

if __name__ == "__main__":
    sys.exit(main())
