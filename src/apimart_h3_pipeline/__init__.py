"""Portable, modular APIMart MiniMax-H3 sequential pipeline."""

from .constants import *
from .media import *
from .policy import *
from .repair import *
from .vision import DashScopeVisionRefiner, observe_stage_output
from .image_editor import GrsaiImageEditor
from .providers.local import LocalH3Client, LocalH3Config, LocalH3Error, LocalH3MediaAdapter
from .bridge import *
from .artifacts import *
from .resources.catalog import camera_motion_prompt, dynamic_action_prompt, image_edit_prompt

__all__ = [name for name in globals() if not name.startswith("_")] + ["main", "parse_args"]


def __getattr__(name: str):
    """Load the orchestration module lazily for ``python -m`` compatibility."""

    if name in {"main", "parse_args"}:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
