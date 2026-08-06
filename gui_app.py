# gui_app.py
"""NetworkSynth's own GUI entry point.

    python gui_app.py

Opens the synthesis window.  It writes a run-spec and launches ``gui_run.py``
as a subprocess, so both entry points share one code path — and when
StructuralGT's button opens this window later, nothing here changes.

On a headless host, relay it over Qt's built-in VNC server:

    QT_QPA_PLATFORM=vnc QT_QUICK_BACKEND=software python gui_app.py

then tunnel with ``ssh -L 5900:localhost:5900 <host>`` and point a VNC client
at ``localhost:5900``.
"""

import sys

from gui.app import main

if __name__ == "__main__":
    sys.exit(main())
