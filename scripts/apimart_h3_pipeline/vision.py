"""Public Qwen-VL API for prompt refinement and observation.

Transport details live in :mod:`vision_client`; prompt orchestration and the
five-frame success gate live in :mod:`vision_refiner`.  The old import path is
kept stable for the runner and downstream experiments.
"""
from __future__ import annotations

from .vision_client import DashScopeClient
from .vision_refiner import DashScopeVisionRefiner, observe_stage_output

__all__ = ["DashScopeClient", "DashScopeVisionRefiner", "observe_stage_output"]
