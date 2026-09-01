"""Backward-compatible export of the DashScope provider boundary."""

from .providers.dashscope_client import DashScopeClient
from .providers.vision_refiner import DashScopeVisionRefiner, observe_stage_output

__all__ = ["DashScopeClient", "DashScopeVisionRefiner", "observe_stage_output"]
