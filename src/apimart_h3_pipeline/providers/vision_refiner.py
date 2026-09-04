"""DashScope Qwen-VL prompt refinement and post-edit observation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .apimart import ApimartError, write_json
from ..core.repair_policy import RepairValidationError, validate_observation

from ..core.constants import OBSERVATION_FRAME_INDICES, PRIMARY_REFERENCE_FRAME_INDEX, QWEN_CONTEXT_FRAME_INDICES
from ..media import select_keyframe
from ..core.policy import camera_motion_kind, normalized_prompt, parse_json_object
from ..resources.catalog import PromptResourceError, image_edit_prompt, render_prompt
from .dashscope_client import DashScopeClient

class DashScopeVisionRefiner(DashScopeClient):
    """Constrained online Qwen-VL refiner for exactly one stage transition."""

    def plan_reference(
        self,
        context_frames: Sequence[Path],
        next_raw_prompt: str,
        is_global_style: bool,
    ) -> dict[str, Any]:
        """Inspect the parent frames while fixing the reference to frame zero."""

        try:
            system = render_prompt(
                "qwen_reference_system.txt",
                primary_frame_index=PRIMARY_REFERENCE_FRAME_INDEX,
            )
            user = render_prompt(
                "qwen_reference_user.txt",
                raw_prompt=next_raw_prompt,
                frame_indices=", ".join(map(str, QWEN_CONTEXT_FRAME_INDICES)),
            )
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        if is_global_style:
            try:
                system += " " + render_prompt("qwen_reference_global_style_suffix.txt")
            except PromptResourceError as error:
                raise ApimartError(str(error)) from error
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self.multimodal_content(user, context_frames)},
            ],
            "temperature": 0,
            "max_tokens": 300,
        }
        response_text, response = self.complete(payload)
        result = parse_json_object(response_text, "Qwen-VL reference planner")
        try:
            selected_frame_index = int(result.get("selected_frame_index"))
        except (TypeError, ValueError) as error:
            raise ApimartError("Qwen-VL returned an invalid selected_frame_index") from error
        if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
            raise ApimartError(
                f"Qwen-VL selected_frame_index must be {PRIMARY_REFERENCE_FRAME_INDEX}: "
                f"{selected_frame_index}"
            )
        usage = response.get("usage")
        # The image editor is deliberately not given Qwen's paraphrase.  An
        # image edit is a pixel-level reference construction step, so adding
        # observed objects (or inferred materials such as "wooden surface")
        # changes content that the atomic request never asked to change. Qwen
        # still selects the frame and records its proposed wording for audit,
        # but the editor receives the raw requirement plus one shared
        # preservation constraint.
        raw_image_edit_prompt = image_edit_prompt(next_raw_prompt)
        return {
            "model": self.model,
            "selected_frame_index": selected_frame_index,
            "selection_reason": normalized_prompt(str(result.get("selection_reason", ""))),
            "image_edit_prompt": raw_image_edit_prompt,
            "image_edit_prompt_source": "raw_atomic_prompt_with_preservation_constraint",
            "frame_observation": normalized_prompt(str(result.get("frame_observation", ""))),
            "is_global_style": is_global_style,
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        }

    def compose_h3_prompt(
        self,
        context_frames: Sequence[Path],
        style_references: Sequence[Path],
        next_raw_prompt: str,
        is_global_style: bool,
        reference_roles: Sequence[Mapping[str, Any]] = (),
        failure_observation: str | None = None,
    ) -> dict[str, Any]:
        """Let Qwen author the final H3 prompt after seeing all actual inputs."""

        picture_count = len(style_references)
        if picture_count not in {0, 1, 3}:
            raise ApimartError(f"unsupported final Qwen picture count: {picture_count}")
        role_lines = [
            f"<Picture {index}> = {str(role.get('role', '')).strip()}, "
            f"source frame {str(role.get('source_frame_index', '')).strip()}"
            for index, role in enumerate(reference_roles, 1)
        ]
        role_contract = "\n".join(role_lines)
        try:
            if picture_count:
                picture_tags = ", ".join(f"<Picture {index}>" for index in range(1, picture_count + 1))
                reference_contract = render_prompt(
                    "qwen_h3_reference_contract.txt",
                    picture_tags=picture_tags,
                    role_contract=role_contract,
                )
            else:
                reference_contract = render_prompt("qwen_h3_no_reference_contract.txt")
            system = render_prompt("qwen_h3_system.txt", reference_contract=reference_contract)
            if is_global_style:
                system += " " + render_prompt("qwen_h3_global_style_suffix.txt")
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        diagnostic = normalized_prompt(failure_observation or "")
        if diagnostic:
            try:
                system += " " + render_prompt("qwen_h3_failure_evidence.txt", failure_observation=diagnostic)
            except PromptResourceError as error:
                raise ApimartError(str(error)) from error
        try:
            user = render_prompt("qwen_h3_user.txt", raw_prompt=next_raw_prompt)
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": self.multimodal_content(
                        user,
                        context_frames,
                        style_references,
                        reference_roles,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 900,
        }
        response_text, response = self.complete(payload)
        # The final editor prompt is intentionally plain text.  JSON is used
        # only by the separate reference-planner and observer control calls.
        h3_prompt = response_text.strip()
        if not h3_prompt:
            raise ApimartError("Qwen-VL returned an empty final H3 prompt")
        usage = response.get("usage")
        return {
            "model": self.model,
            "h3_prompt": h3_prompt,
            "h3_prompt_source": "qwen_vl_direct",
            "frame_observation": "",
            "picture_count": picture_count,
            "is_global_style": is_global_style,
            "repair_attempts": [],
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        }

    def observe(
        self,
        frames: Sequence[Path],
        atomic_prompt: str,
        source_frames: Sequence[Path] = (),
    ) -> dict[str, Any]:
        """Judge one atomic edit from paired source/output checkpoints."""

        if len(frames) != len(OBSERVATION_FRAME_INDICES):
            raise ApimartError(
                "Qwen-VL success gate requires exactly five observation frames: "
                f"{len(frames)} != {len(OBSERVATION_FRAME_INDICES)}"
            )
        if source_frames and len(source_frames) != len(OBSERVATION_FRAME_INDICES):
            raise ApimartError(
                "Qwen-VL source comparison requires exactly five source frames: "
                f"{len(source_frames)} != {len(OBSERVATION_FRAME_INDICES)}"
            )
        try:
            system = render_prompt("qwen_observer_system.txt")
            user = render_prompt("qwen_observer_user.txt", raw_prompt=atomic_prompt)
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        content: list[dict[str, Any]] = [{"type": "text", "text": system + "\n\n" + user}]
        if source_frames:
            for frame_index, frame in zip(OBSERVATION_FRAME_INDICES, source_frames, strict=True):
                content.append({"type": "text", "text": f"Source frame {frame_index}:"})
                content.append({"type": "image_url", "image_url": {"url": self.image_data_url(frame)}})
        for frame_index, frame in zip(OBSERVATION_FRAME_INDICES, frames, strict=True):
            content.append({"type": "text", "text": f"Generated output frame {frame_index}:"})
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(frame)}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 300,
        }
        content_text, response = self.complete(payload)
        result = parse_json_object(content_text, "Qwen-VL success gate")
        try:
            normalized_result = validate_observation(result)
        except RepairValidationError as error:
            raise ApimartError(str(error)) from error
        usage = response.get("usage")
        return {
            "success": normalized_result["success"],
            "failure_type": normalized_result["failure_type"],
            "observation": normalized_result["observation"],
            "observer_evidence": normalized_result["observer_evidence"],
            "confidence": normalized_result["confidence"],
            "model": self.model,
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        }


def _camera_translation_check(
    source_frames: Sequence[Path],
    output_frames: Sequence[Path],
    atomic_prompt: str,
) -> dict[str, Any] | None:
    """Measure global horizontal displacement for a camera-motion stage.

    Qwen can mistake a static or regenerated frame sequence for a successful
    pan.  Phase correlation is deliberately conservative: require a coherent
    multi-frame displacement of at least four pixels before accepting the
    requested camera motion as visibly present.
    """

    if not source_frames or len(source_frames) != len(output_frames):
        return None
    if camera_motion_kind(atomic_prompt) is None:
        return None
    shifts: list[float] = []
    responses: list[float] = []
    for source, output in zip(source_frames, output_frames, strict=True):
        source_image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        output_image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        if source_image is None or output_image is None or source_image.shape != output_image.shape:
            return {"ok": False, "reason": "camera_check_missing_or_mismatched_frame"}
        source_image = source_image.astype(np.float32)
        output_image = output_image.astype(np.float32)
        height, width = source_image.shape
        # Ignore a narrow border where codec padding and edge synthesis can
        # dominate the correlation peak.
        margin_x = max(8, width // 32)
        margin_y = max(4, height // 32)
        shift, response = cv2.phaseCorrelate(
            source_image[margin_y:-margin_y, margin_x:-margin_x],
            output_image[margin_y:-margin_y, margin_x:-margin_x],
        )
        shifts.append(float(shift[0]))
        responses.append(float(response))
    median_shift = float(np.median(shifts))
    coherent = sum(abs(value) >= 4.0 for value in shifts) >= max(3, len(shifts) // 2 + 1)
    ok = coherent and abs(median_shift) >= 4.0
    return {
        "ok": ok,
        "median_horizontal_shift_px": round(median_shift, 3),
        "horizontal_shifts_px": [round(value, 3) for value in shifts],
        "correlation_responses": [round(value, 3) for value in responses],
    }


def observe_stage_output(
    refiner: DashScopeVisionRefiner,
    output: Path,
    stage_dir: Path,
    task_id: str,
    stage_label: str,
    atomic_prompt: str,
    source_video: Path | None = None,
) -> dict[str, Any]:
    """Extract paired source/output checkpoints and persist the success gate."""

    observation_dir = stage_dir / "observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"task_{task_id}_{stage_label}_output_observation_frame"
    frames: list[Path] = []
    source_frames: list[Path] = []
    if source_video is not None:
        for frame_index in OBSERVATION_FRAME_INDICES:
            source_frame = observation_dir / f"{file_prefix.replace('_output_', '_source_')}_{frame_index:03d}.png"
            select_keyframe(source_video, source_frame, frame_index)
            source_frames.append(source_frame)
    for frame_index in OBSERVATION_FRAME_INDICES:
        frame = observation_dir / f"{file_prefix}_{frame_index:03d}.png"
        select_keyframe(output, frame, frame_index)
        frames.append(frame)
    try:
        result = validate_observation(refiner.observe(frames, atomic_prompt, source_frames))
        camera_check = _camera_translation_check(source_frames, frames, atomic_prompt)
        if camera_check is not None and not camera_check.get("ok"):
            result = {
                **result,
                "success": False,
                "failure_type": "motion_weak",
                "observation": (
                    "Local camera-motion check found no coherent horizontal displacement "
                    f"in the source/output pairs: {camera_check}"
                ),
                "observer_evidence": (
                    "Local camera-motion check found no coherent horizontal displacement "
                    f"in the source/output pairs: {camera_check}"
                ),
                "confidence": max(float(result.get("confidence", 0.0)), 0.95),
                "camera_motion_check": camera_check,
            }
        elif camera_check is not None:
            result = {**result, "camera_motion_check": camera_check}
    except (ApimartError, RepairValidationError) as error:
        # A Qwen transport failure is not evidence that H3 failed. Record it
        # separately so the repair policy never treats it as a semantic failure.
        result = {
            "success": None,
            "failure_type": "observer_unavailable",
            "observation": "observer_unavailable: " + str(error),
            "observer_evidence": "observer_unavailable: " + str(error),
            "confidence": 0.0,
            "model": refiner.model,
            "error": str(error),
        }
    record = {
        "kind": "qwen_vl_five_frame_success_gate_v1",
        "stage": stage_label,
        "task_id": task_id,
        "frame_indices": list(OBSERVATION_FRAME_INDICES),
        "frames": [str(frame) for frame in frames],
        "source_frames": [str(frame) for frame in source_frames],
        "atomic_prompt": atomic_prompt,
        **result,
    }
    write_json(observation_dir / "observation.json", record)
    print(json.dumps({
        "event": "post_edit_observation",
        "stage": stage_label,
        "success": result.get("success"),
        "confidence": result.get("confidence"),
        "observation": result.get("observation", ""),
        "frame_indices": list(OBSERVATION_FRAME_INDICES),
    }, ensure_ascii=False), flush=True)
    return record
