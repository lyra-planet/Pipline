"""Load pipeline prompt templates from package resources.

Prompt text is kept outside Python so it can be reviewed and edited as a
normal artifact.  Callers persist the prompts they already use as part of
their existing bridge and attempt artifacts.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any


class PromptResourceError(RuntimeError):
    """Raised when a prompt resource is missing or malformed."""


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PromptResourceError("prompt resource name must be a non-empty string")
    candidate = Path(name)
    if candidate.name != name or name in {".", ".."} or candidate.is_absolute():
        raise PromptResourceError(f"invalid prompt resource name: {name!r}")
    return name


def _resource(name: str):
    safe_name = _validate_name(name)
    resource = resources.files("apimart_h3_pipeline").joinpath("resources", "prompts", safe_name)
    if not resource.is_file():
        raise PromptResourceError(f"prompt resource does not exist: {safe_name}")
    return resource


def load_prompt(name: str) -> str:
    """Return UTF-8 prompt text without depending on the current directory."""

    try:
        return _resource(name).read_text(encoding="utf-8")
    except PromptResourceError:
        raise
    except (OSError, UnicodeError) as error:
        raise PromptResourceError(f"could not read prompt resource {name!r}") from error


def render_prompt(name: str, **values: Any) -> str:
    """Render a ``string.Template`` prompt and reject missing placeholders."""

    try:
        rendered = Template(load_prompt(name)).substitute(values)
    except KeyError as error:
        raise PromptResourceError(
            f"prompt resource {name!r} requires missing value {error.args[0]!r}"
        ) from error
    except ValueError as error:
        raise PromptResourceError(f"prompt resource {name!r} has invalid template syntax") from error
    if "${" in rendered:
        raise PromptResourceError(f"prompt resource {name!r} contains an unresolved placeholder")
    return rendered.strip()


def image_edit_prompt(raw_prompt: str) -> str:
    """Add the shared preservation constraint to an image-edit request.

    The atomic requirement remains the source of all requested changes.  This
    clause only tells the image editor not to redesign unrelated content in
    the reference frame.
    """

    requirement = " ".join(str(raw_prompt).split()).strip()
    if not requirement:
        raise PromptResourceError("image edit prompt requires a non-empty atomic requirement")
    preservation = load_prompt("image_edit_preservation.txt").strip()
    if preservation.lower() in requirement.lower():
        return requirement
    return f"{requirement} {preservation}"


def camera_motion_prompt(raw_prompt: str) -> str:
    """Append only the physical-camera clarification for the named movement."""

    requirement = " ".join(str(raw_prompt).split()).strip()
    if not requirement:
        raise PromptResourceError("camera motion prompt requires a non-empty atomic requirement")
    # Imported lazily to keep the policy module independent from resources.
    from ..core.policy import camera_motion_kind

    motion_kind = camera_motion_kind(requirement)
    if motion_kind is None:
        return requirement
    try:
        clauses = json.loads(load_prompt("h3_camera_motion_clauses.json"))
    except json.JSONDecodeError as error:
        raise PromptResourceError("h3_camera_motion_clauses.json is not valid JSON") from error
    if not isinstance(clauses, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in clauses.items()
    ):
        raise PromptResourceError(
            "h3_camera_motion_clauses.json must map movement names to non-empty strings"
        )
    common = load_prompt("h3_camera_motion_contract.txt").strip()
    specific = str(clauses.get(motion_kind, clauses.get("generic", ""))).strip()
    if not specific:
        raise PromptResourceError(f"camera motion resource lacks clause for {motion_kind!r}")
    contract = f"{specific} {common}".strip()
    if contract.lower() in requirement.lower():
        return requirement
    return f"{requirement} {contract}"


def dynamic_action_prompt(raw_prompt: str) -> str:
    """Append the temporal-onset contract for a pure action/pose edit."""

    requirement = " ".join(str(raw_prompt).split()).strip()
    if not requirement:
        raise PromptResourceError("dynamic action prompt requires a non-empty atomic requirement")
    contract = load_prompt("h3_dynamic_action_contract.txt").strip()
    if contract.lower() in requirement.lower():
        return requirement
    return f"{requirement} {contract}"


def load_repair_clauses() -> dict[str, str]:
    """Load the closed repair-action clause table from package data."""

    try:
        value = json.loads(load_prompt("repair_clauses.json"))
    except json.JSONDecodeError as error:
        raise PromptResourceError("repair_clauses.json is not valid JSON") from error
    if not isinstance(value, dict) or not value or any(
        not isinstance(key, str) or not key.strip() or not isinstance(clause, str) or not clause.strip()
        for key, clause in value.items()
    ):
        raise PromptResourceError("repair_clauses.json must map non-empty action names to non-empty strings")
    return {key: clause.strip() for key, clause in value.items()}


__all__ = [
    "PromptResourceError",
    "load_prompt",
    "render_prompt",
    "image_edit_prompt",
    "camera_motion_prompt",
    "dynamic_action_prompt",
    "load_repair_clauses",
]
