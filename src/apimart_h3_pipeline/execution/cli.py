"""Command-line argument definitions for the sequential H3 runner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..core.constants import (
    DASHSCOPE_DEFAULT_BASE_URL,
    DASHSCOPE_DEFAULT_MODEL,
    DEFAULT_STATIC_REFERENCE_COUNT,
    REFERENCE_IMAGE_COUNTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-jobs", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-public-base-url", required=True)
    parser.add_argument(
        "--prepared-initial-video",
        type=Path,
        help="reuse a letterboxed 1344x768 task input and its .geometry.json sidecar",
    )
    parser.add_argument(
        "--apimart-env", type=Path,
        default=Path(os.environ.get("APIMART_ENV_FILE", "~/.apimart.env")).expanduser(),
        help="APIMart credentials file (or APIMART_ENV_FILE)",
    )
    parser.add_argument(
        "--grsai-env", type=Path,
        default=Path(os.environ.get("GRSAI_ENV_FILE", "~/.grsai.env")).expanduser(),
        help="GRSAI credentials file (or GRSAI_ENV_FILE)",
    )
    parser.add_argument(
        "--dashscope-env", type=Path,
        default=Path(os.environ.get("DASHSCOPE_ENV_FILE", "~/.dashscope.env")).expanduser(),
        help="DashScope credentials file (or DASHSCOPE_ENV_FILE)",
    )
    parser.add_argument("--dashscope-base-url", default=DASHSCOPE_DEFAULT_BASE_URL)
    parser.add_argument("--dashscope-model", default=DASHSCOPE_DEFAULT_MODEL)
    parser.add_argument("--dashscope-timeout", type=int, default=120)
    parser.add_argument("--h3-model", default="MiniMax-H3")
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--resolution", choices=("768P", "2K"), default="768P")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--poll-seconds", type=float, default=7.0)
    parser.add_argument("--total-timeout", type=int, default=900)
    parser.add_argument("--allow-resubmit", action="store_true", help="retry only a state already proven not to have created an API task")
    parser.add_argument("--last-stage", help="stop after this inclusive stage ID, for example S3")
    parser.add_argument(
        "--initial-reference", action="store_true",
        help="legacy compatibility flag; every stage now uses the same bridge contract",
    )
    parser.add_argument(
        "--global-style-reference-count", type=int, choices=REFERENCE_IMAGE_COUNTS, default=DEFAULT_STATIC_REFERENCE_COUNT,
        help="reference count for ordinary static bridges; global styles use one anchor first and three only on failed retry",
    )
    parser.add_argument(
        "--failure-recovery",
        choices=("targeted", "fixed-three-anchor", "disabled"),
        default="targeted",
        help="semantic failure recovery policy: diagnose targeted repair, legacy fixed anchors, or stop",
    )
    parser.add_argument(
        "--allow-unverified-output",
        action="store_true",
        help="legacy mode: allow an unavailable Observer to propagate media without semantic confirmation",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
