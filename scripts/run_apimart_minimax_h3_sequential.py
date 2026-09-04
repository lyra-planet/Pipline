#!/usr/bin/env python3
"""Compatibility entry point for the modular APIMart MiniMax-H3 pipeline.

The implementation lives in :mod:`apimart_h3_pipeline`.  This file remains
at the historical path so existing screen wrappers, notebooks, and imports do
not need to change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline import *  # noqa: F401,F403
from apimart_h3_pipeline.execution import artifacts as _artifacts
from apimart_h3_pipeline import bridge as _bridge
from apimart_h3_pipeline.providers import grsai as _image_editor
from apimart_h3_pipeline.providers.grsai import aspect_ratio_for_image
from apimart_h3_pipeline.media import video as _media
from apimart_h3_pipeline.core import policy as _policy
from apimart_h3_pipeline.execution import runner as _runner
from apimart_h3_pipeline.providers import vision_refiner as _vision

from apimart_h3_pipeline.providers.apimart import ApimartClient, ApimartError, resolve_credentials, write_json
from apimart_h3_pipeline.core.repair_policy import (
    FailureDiagnosisAndRepair,
    RepairValidationError,
    STAGE_RETRY_LIMIT,
    fixed_three_anchor_repair,
    global_style_three_anchor_repair,
    stage_outcome,
    validate_observation,
    validate_repair_record,
)


def _sync_compatibility_patches() -> None:
    """Forward legacy-module monkey patches into the modular runner.

    Tests and downstream scripts historically patched symbols on this module.
    Keeping this tiny bridge avoids a silent change in dependency injection
    semantics while making the production implementation modular.
    """

    modules = (_runner, _artifacts, _bridge, _image_editor, _media, _policy, _vision)
    names = (
        "ApimartClient", "ApimartError", "DashScopeVisionRefiner", "FailureDiagnosisAndRepair",
        "GrsaiImageEditor", "RepairValidationError", "archive_stage_attempt", "bridge_for_stage",
        "compute_canvas_geometry", "confirmed_previous_requirements", "deterministic_repair_h3_prompt",
        "fixed_three_anchor_repair", "global_style_three_anchor_repair", "has_audio", "invoke_h3_client",
        "is_aligned_video", "is_h3_generated_video", "is_h3_input_video", "load_geometry_sidecar",
        "load_task", "materialize_final_video", "materialize_initial_video", "materialize_stage_video",
        "observe_stage_output", "parse_args", "public_url", "read_json", "reference_policy",
        "replace_manifest_stage", "resolve_credentials", "reusable_h3_video_url", "source_canvas_geometry",
        "select_keyframe", "stage_failure_entry", "stage_outcome", "three_anchor_reference_plan", "validate_observation",
        "validate_repair_record", "write_geometry_sidecar", "write_json",
    )
    for name in names:
        if name not in globals():
            continue
        value = globals()[name]
        for module in modules:
            if name == "bridge_for_stage" and module is not _runner:
                continue
            if hasattr(module, name):
                setattr(module, name, value)
        if hasattr(_runner, name):
            setattr(_runner, name, value)


def bridge_for_stage(*args, **kwargs):
    """Compatibility wrapper that keeps patched frame extraction effective."""

    _bridge.select_keyframe = globals().get("select_keyframe", _media.select_keyframe)
    return _bridge.bridge_for_stage(*args, **kwargs)


def main() -> int:
    """Run the modular runner while preserving legacy patch points."""

    _sync_compatibility_patches()
    return _runner.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
