"""Small, reusable pieces of the reference-image bridge.

The stage bridge itself coordinates the execution flow.  This module owns the
pure task/URL helpers and the deterministic reference and repair prompt
contracts so those rules can be tested without running a provider request.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..providers.apimart import ApimartClient, ApimartError
from ..core.repair_policy import RepairValidationError, apply_repair_clause

from ..execution.artifacts import attempt_number_from_path
from ..core.constants import (
    BRIDGE_KIND,
    GLOBAL_STYLE_REFERENCE_COUNT,
    PRIMARY_REFERENCE_FRAME_INDEX,
    PROMPT_KEY,
    TEMPORAL_END_FRAME_INDEX,
    TEMPORAL_MIDDLE_FRAME_INDEX,
)
from ..media import read_json
from ..core.policy import expected_reference_roles, is_camera_motion_edit, is_dynamic_action_edit, normalized_prompt
from ..resources.catalog import PromptResourceError, camera_motion_prompt, dynamic_action_prompt, image_edit_prompt, render_prompt


def load_task(compiled_jobs: Path, task_id: str) -> dict[str, Any]:
    document = read_json(compiled_jobs)
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ApimartError("compiled jobs document lacks tasks list")
    for task in tasks:
        if isinstance(task, Mapping) and str(task.get("task_id")) == task_id:
            stages = task.get("sequential_nominal_plan")
            source = Path(str(task.get("source_video", ""))).resolve()
            if not isinstance(stages, list) or not stages or not source.is_file():
                raise ApimartError(f"compiled task {task_id} is malformed")
            checked = []
            for index, stage in enumerate(stages, 1):
                if not isinstance(stage, Mapping) or str(stage.get("stage_id")) != f"S{index}":
                    raise ApimartError(f"compiled task {task_id} has non-sequential stages")
                prompt = str(stage.get(PROMPT_KEY, "")).strip()
                if not prompt:
                    raise ApimartError(f"compiled task {task_id} stage S{index} lacks raw content-only prompt")
                checked.append({"stage_id": f"S{index}", "prompt": prompt})
            return {"task_id": task_id, "source_video": source, "stages": checked}
    raise ApimartError(f"task_id {task_id} is absent from compiled jobs")


def public_url(base_url: str, filename: str) -> str:
    return base_url.rstrip("/") + "/" + urllib.parse.quote(filename)


def upload_reference_image(apimart: ApimartClient, image: Path) -> dict[str, Any]:
    """Retry transport-only image uploads without repeating the image edit."""

    last_error: ApimartError | None = None
    for attempt in range(1, 4):
        try:
            uploaded = apimart.upload_image(image)
            url = uploaded.get("url")
            if isinstance(url, str):
                parsed = urllib.parse.urlsplit(url)
                uploaded["url"] = urllib.parse.urlunsplit((
                    parsed.scheme,
                    parsed.netloc,
                    urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%:@-._~!$&'()*+,;="),
                    parsed.query,
                    parsed.fragment,
                ))
            return uploaded
        except ApimartError as error:
            last_error = error
            if attempt == 3:
                break
            print(json.dumps({
                "event": "reference_image_upload_retry",
                "attempt": attempt,
                "error": str(error),
            }, ensure_ascii=False), flush=True)
            time.sleep(attempt * 3)
    raise ApimartError("reference image upload failed after 3 attempts") from last_error


def relocated_artifact(path_value: Any, relocated_dir: Path) -> Path | None:
    """Resolve an artifact after its bridge directory has been archived."""

    if not isinstance(path_value, str) or not path_value.strip():
        return None
    original = Path(path_value)
    for candidate in (original, relocated_dir / original.name):
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def prior_primary_reference(
    stage_dir: Path,
    previous_video: Path,
    next_prompt: str,
    refiner_model: str,
) -> tuple[dict[str, Any], Path] | None:
    """Find the latest edited primary frame moved under an archived attempt."""

    attempts_root = stage_dir / "attempts"
    attempt_dirs = sorted(
        (
            item for item in attempts_root.iterdir()
            if item.is_dir() and attempt_number_from_path(item) is not None
        ),
        key=lambda item: attempt_number_from_path(item) or 0,
        reverse=True,
    ) if attempts_root.is_dir() else []
    for attempt_dir in attempt_dirs:
        archived_bridge_dir = attempt_dir / "bridge_for_next"
        archived_bridge_path = archived_bridge_dir / "bridge.json"
        if not archived_bridge_path.is_file():
            continue
        bridge = read_json(archived_bridge_path)
        reference_plan = bridge.get("reference_plan", bridge.get("refiner"))
        if not (
            bridge.get("kind") == BRIDGE_KIND
            and bridge.get("stage_id", stage_dir.name) == stage_dir.name
            and bridge.get("previous_video") == str(previous_video)
            and bridge.get("next_raw_prompt") == next_prompt
            and isinstance(reference_plan, Mapping)
            and reference_plan.get("model") == refiner_model
        ):
            continue
        try:
            selected_frame_index = int(
                bridge.get("selected_frame_index", reference_plan.get("selected_frame_index"))
            )
        except (TypeError, ValueError) as error:
            raise ApimartError(f"{attempt_dir.name} bridge lacks Qwen's selected frame index") from error
        if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
            continue
        reference_images = bridge.get("reference_images")
        if not isinstance(reference_images, list) or not reference_images:
            continue
        primary_index: int | None = None
        roles = bridge.get("reference_roles")
        if isinstance(roles, list):
            for index, role in enumerate(roles):
                if (
                    isinstance(role, Mapping)
                    and role.get("source_frame_index") == PRIMARY_REFERENCE_FRAME_INDEX
                    and role.get("role") in {"edited start anchor", "edited primary anchor"}
                ):
                    primary_index = index
                    break
        if primary_index is None and len(reference_images) == 1:
            primary_index = 0
        if primary_index is None or primary_index >= len(reference_images):
            continue
        primary_reference = relocated_artifact(reference_images[primary_index], archived_bridge_dir)
        if primary_reference is None:
            raise ApimartError(f"{attempt_dir.name} primary reference image is missing after archival")
        return dict(reference_plan), primary_reference
    return None


def upload_bridge_image(
    apimart: ApimartClient,
    image: Path,
    media_public_base_url: str | None,
    media_dir: Path | None,
) -> dict[str, Any]:
    """Expose one generated reference through the provider's supported route."""

    if apimart.is_ctmoai:
        return {"url": apimart.upload_media(image, "images"), "upload_mode": "ctmoai_sd_media"}
    if media_public_base_url and media_dir is not None:
        media_dir.mkdir(parents=True, exist_ok=True)
        media_image = media_dir / image.name
        if image.resolve() != media_image.resolve():
            shutil.copy2(image, media_image)
        return {
            "url": public_url(media_public_base_url, media_image.name),
            "filename": media_image.name,
            "bytes": media_image.stat().st_size,
            "upload_mode": "local_public_media_server",
        }
    return upload_reference_image(apimart, image)


def temporal_anchor_h3_prompt(
    raw_prompt: str,
    reference_roles: Sequence[Mapping[str, Any]],
) -> str:
    """Build the stable prompt contract for a three-image appearance edit."""

    requirement = normalized_prompt(raw_prompt)
    if not requirement:
        raise ApimartError("three-anchor H3 prompt requires a non-empty atomic requirement")
    if len(reference_roles) != GLOBAL_STYLE_REFERENCE_COUNT:
        raise ApimartError(
            "three-anchor H3 prompt requires exactly "
            f"{GLOBAL_STYLE_REFERENCE_COUNT} temporal roles"
        )
    anchor_lines: list[str] = []
    for picture_index, role in enumerate(reference_roles, 1):
        role_name = normalized_prompt(str(role.get("role", "")))
        source_frame = role.get("source_frame_index")
        if not role_name or not isinstance(source_frame, int):
            raise ApimartError(f"invalid temporal role for Picture {picture_index}")
        anchor_lines.append(f"Use <Picture {picture_index}> as the {role_name} at source frame {source_frame}.")
    try:
        prompt = render_prompt(
            "h3_temporal_anchor.txt",
            requirement=requirement,
            anchor_lines=" ".join(anchor_lines),
        )
    except PromptResourceError as error:
        raise ApimartError(str(error)) from error
    return prompt


def three_anchor_reference_plan(
    model: str,
    raw_prompt: str,
    selected_frame_index: int,
    is_global_style: bool,
) -> dict[str, Any]:
    """Describe deterministic temporal-anchor edits without another model call."""

    requirement = normalized_prompt(raw_prompt)
    if not requirement:
        raise ApimartError("three-anchor reference plan requires a non-empty atomic requirement")
    if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
        raise ApimartError(
            "three-anchor reference plan requires the first parent frame as its style master: "
            f"{selected_frame_index} != {PRIMARY_REFERENCE_FRAME_INDEX}"
        )
    return {
        "model": model,
        "selected_frame_index": selected_frame_index,
        "style_reference_frame_index": PRIMARY_REFERENCE_FRAME_INDEX,
        "middle_frame_index": TEMPORAL_MIDDLE_FRAME_INDEX,
        "end_frame_index": TEMPORAL_END_FRAME_INDEX,
        "middle_image_edit_prompt": image_edit_prompt(requirement),
        "end_image_edit_prompt": image_edit_prompt(requirement),
        "image_edit_prompt_source": "raw_atomic_prompt_with_preservation_constraint",
        "is_global_style": is_global_style,
        "frame_observation": (
            "the first-frame primary is the shared style master; middle and end anchors "
            "are edited from their own parent frames"
        ),
        "usage": {},
    }


def deterministic_repair_h3_prompt(
    raw_prompt: str,
    picture_count: int,
    reference_roles: Sequence[Mapping[str, Any]],
    repair_context: Mapping[str, Any],
) -> str:
    """Build a repair prompt without another free-form Qwen rewrite."""

    if picture_count not in {0, 1, 3}:
        raise ApimartError(f"unsupported repair reference count: {picture_count}")
    try:
        if picture_count == 0:
            requirement = (
                dynamic_action_prompt(raw_prompt)
                if is_dynamic_action_edit(raw_prompt)
                else raw_prompt.strip()
            )
            base = render_prompt("h3_video_only_repair.txt", requirement=requirement)
        elif picture_count == 1:
            base = render_prompt("h3_one_anchor_repair.txt", requirement=raw_prompt.strip())
        else:
            if len(reference_roles) != 3:
                raise ApimartError("three-anchor repair requires exactly three reference roles")
            base = temporal_anchor_h3_prompt(raw_prompt, reference_roles)
    except PromptResourceError as error:
        raise ApimartError(str(error)) from error
    try:
        repaired = apply_repair_clause(
            base,
            current_requirement=raw_prompt,
            repair=repair_context,
            picture_count=picture_count,
        )
        if is_camera_motion_edit(raw_prompt):
            repaired = camera_motion_prompt(repaired)
        return repaired
    except RepairValidationError as error:
        raise ApimartError(str(error)) from error


__all__ = [
    "load_task",
    "public_url",
    "upload_reference_image",
    "relocated_artifact",
    "prior_primary_reference",
    "upload_bridge_image",
    "temporal_anchor_h3_prompt",
    "three_anchor_reference_plan",
    "deterministic_repair_h3_prompt",
]
