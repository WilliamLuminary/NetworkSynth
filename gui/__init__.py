# src/gui/__init__.py
"""NetworkSynth's own GUI.

Two entry points share everything below the surface:

* ``gui_app.py``  — this window, run directly
* ``gui_run.py``  — a run-spec on the command line

The window does not call the pipelines in-process: it writes a run-spec, starts
``gui_run.py`` as a subprocess, and follows ``run.jsonl`` for progress.  That is
the same mechanism StructuralGT's controller will use, so this window doubles as
its reference implementation.
"""
