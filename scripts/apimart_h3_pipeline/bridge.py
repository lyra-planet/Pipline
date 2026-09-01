"""Public bridge API for the sequential H3 pipeline.

The implementation is split between deterministic bridge helpers and the
stage execution coordinator.  This module keeps the original import surface
stable for launchers and tests.
"""
from __future__ import annotations

from . import bridge_execution as _execution
from .bridge_helpers import (
    deterministic_repair_h3_prompt,
    load_task,
    prior_primary_reference,
    public_url,
    relocated_artifact,
    temporal_anchor_h3_prompt,
    three_anchor_reference_plan,
    upload_bridge_image,
    upload_reference_image,
)
from .media import select_keyframe


def bridge_for_stage(*args, **kwargs):
    """Run the coordinator while preserving the historical patch point."""

    _execution.select_keyframe = select_keyframe
    return _execution.bridge_for_stage(*args, **kwargs)

__all__ = [
    "bridge_for_stage",
    "load_task",
    "public_url",
    "upload_reference_image",
    "relocated_artifact",
    "prior_primary_reference",
    "upload_bridge_image",
    "temporal_anchor_h3_prompt",
    "three_anchor_reference_plan",
    "deterministic_repair_h3_prompt",
    "select_keyframe",
]
