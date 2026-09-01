#!/usr/bin/env python3
"""Compatibility command for the packaged APIMart H3 provider client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.providers.apimart import *  # noqa: F401,F403,E402
from apimart_h3_pipeline.providers.apimart import ApimartError, main  # noqa: E402


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
