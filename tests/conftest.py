"""Test bootstrap: import the package from ``src/`` without installing it.

Deliberate: CI must be able to run the CPU suite from a bare checkout (``pytest``), and an editable
install would hide a broken ``packages.find``/layout until release time.
"""

from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLATFORM_SRC = SRC.parent.parent / "AscendPlatform" / "src"
