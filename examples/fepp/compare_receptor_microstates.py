#!/usr/bin/env python3
"""Compare prepared receptors using the packaged standalone Suite worker.

Invoke with ``$SCHRODINGER/run python3 compare_receptor_microstates.py``.
The worker is loaded by file path so Suite Python 3.11 need not import mdpp,
which requires Python 3.12 or newer.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_WORKER = Path(__file__).resolve().parents[2] / "src/mdpp/prep/_fepp_receptor_worker.py"


if __name__ == "__main__":
    runpy.run_path(str(_WORKER), run_name="__main__")
