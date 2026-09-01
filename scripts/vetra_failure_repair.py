"""Compatibility import for the packaged VETRA repair policy."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.core.repair_policy import *  # noqa: F401,F403,E402
