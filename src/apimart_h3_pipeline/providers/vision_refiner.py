"""DashScope Qwen-VL prompt refinement and post-edit observation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .apimart import ApimartError, write_json
from ..core.repair_policy import RepairValidationError, validate_observation

from ..core.constants import (
    POST_EDIT_OBSERVER_FRAME_INDICES,
    PRIMARY_REFERENCE_FRAME_INDEX,
    QWEN_CONTEXT_FRAME_INDICES,
)
from ..media import select_keyframe
from ..core.policy import normalized_prompt, parse_json_object
from ..resources.catalog import PromptResourceError, image_edit_prompt, render_prompt
from .dashscope_client import DashScopeClient


# Bump this whenever the instructions that author a fallback final H3 prompt
# change. Existing bridges with an older value must be regenerated so a
# resumed fallback cannot silently reuse a weak prompt.
FAILURE_REPAIR_PROMPT_VERSION = "qwen_h3_failure_repair_user_v5_no_meta_preservation"


class DashScopeVisionRefiner(DashScopeClient):
    """Constrained online Qwen-VL refiner for exactly one stage transition."""

    def plan_reference(
        self,
        context_frames: Sequence[Path],
        next_raw_prompt: str,
        is_global_style: bool,
    ) -> dict[str, Any]:
        """Inspect all parent frames and choose the clearest editable anchor."""

        try:
            system = render_prompt(
                "qwen_reference_system.txt",
                primary_frame_index=PRIMARY_REFERENCE_FRAME_INDEX,
                candidate_frame_indices=", ".join(map(str, QWEN_CONTEXT_FRAME_INDICES)),
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
        if selected_frame_index not in QWEN_CONTEXT_FRAME_INDICES:
            raise ApimartError(
                "Qwen-VL selected_frame_index must be one of "
                f"{QWEN_CONTEXT_FRAME_INDICES}: "
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
        failed_h3_prompt: str | None = None,
        repair_action: str | None = None,
        failure_type: str | None = None,
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
        is_failure_repair = bool(failed_h3_prompt or repair_action)
        if is_failure_repair:
            try:
                system += " " + render_prompt(
                    "qwen_h3_failure_repair_contract.txt",
                    failure_observation="the quoted Observer evidence in the user message",
                    repair_action="the quoted repair action hint in the user message",
                )
            except PromptResourceError as error:
                raise ApimartError(str(error)) from error
        try:
            if is_failure_repair:
                user = render_prompt(
                    "qwen_h3_failure_repair_user.txt",
                    raw_prompt=normalized_prompt(next_raw_prompt),
                    failed_h3_prompt=normalized_prompt(failed_h3_prompt or "not supplied"),
                    failure_observation=diagnostic or "not supplied",
                    failure_type=normalized_prompt(failure_type or "not supplied"),
                    repair_action=normalized_prompt(repair_action or "not supplied"),
                )
            else:
                user = render_prompt("qwen_h3_user.txt", raw_prompt=next_raw_prompt)
            if is_failure_repair:
                system = render_prompt("qwen_h3_fallback_final_system.txt")
                user = render_prompt("qwen_h3_fallback_final_user.txt", raw_prompt=next_raw_prompt, failed_h3_prompt=failed_h3_prompt or "not supplied", failure_observation=diagnostic, role_contract=role_contract if picture_count else "No reference pictures are attached.")
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
            "raw_atomic_prompt": normalized_prompt(next_raw_prompt),
            "raw_prompt_provenance": "authoritative_atomic_requirement_in_final_qwen_user_message",
            "frame_observation": "",
            "picture_count": picture_count,
            "is_global_style": is_global_style,
            "failure_repair_prompt_version": (
                FAILURE_REPAIR_PROMPT_VERSION if is_failure_repair else None
            ),
            "failure_type": normalized_prompt(failure_type or "") if is_failure_repair else None,
            "repair_attempts": [],
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        }

    def plan_failure_repair(
        self,
        context_frames: Sequence[Path],
        existing_references: Sequence[Path],
        raw_prompt: str,
        failed_h3_prompt: str,
        failure_type: str,
        failure_observation: str,
        reference_roles: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Ask Qwen to choose repair reference topology and per-anchor edits."""

        if len(context_frames) != len(QWEN_CONTEXT_FRAME_INDICES):
            raise ApimartError("fallback repair planning requires five source frames")
        try:
            system = render_prompt("qwen_failure_repair_system.txt")
            user = render_prompt(
                "qwen_failure_repair_user.txt",
                raw_prompt=raw_prompt,
                failed_h3_prompt=failed_h3_prompt,
                failure_type=failure_type,
                failure_observation=failure_observation,
                frame_indices=", ".join(map(str, QWEN_CONTEXT_FRAME_INDICES)),
            )
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        role_text = "".join(
            f"\nExisting Picture {i}: {role.get('role', '')}, source frame {role.get('source_frame_index', '')}."
            for i, role in enumerate(reference_roles, 1)
        )
        if role_text:
            user += role_text
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {
                "role": "user",
                "content": self.multimodal_content(
                    user,
                    context_frames,
                    existing_references,
                    reference_roles,
                ),
            }],
            "temperature": 0,
            "max_tokens": 1200,
        }
        response_text, response = self.complete(payload)
        result = parse_json_object(response_text, "Qwen-VL failure repair planner")
        # Give the model one corrective pass when its topology or per-image
        # frame mapping is malformed. This is a prompt-level correction, not
        # a local decision about whether references are useful.
        def valid_reference_plan(candidate: Mapping[str, Any]) -> bool:
            candidate_policy = str(candidate.get("reference_policy", "")).strip().lower()
            candidate_indices = candidate.get("anchor_frame_indices", [])
            candidate_prompts = candidate.get("image_edit_prompts", [])
            if not isinstance(candidate_indices, list) or not isinstance(candidate_prompts, list):
                return False
            if candidate_policy == "video_only":
                return not candidate_indices and not candidate_prompts
            if candidate_policy == "one_anchor":
                return (
                    len(candidate_indices) == 1
                    and len(candidate_prompts) == 1
                    and isinstance(candidate_prompts[0], Mapping)
                    and candidate_prompts[0].get("frame_index") == candidate_indices[0]
                )
            if candidate_policy == "three_anchor":
                return (
                    len(candidate_indices) == 3
                    and all(isinstance(i, int) and not isinstance(i, bool) for i in candidate_indices)
                    and candidate_indices[0] == QWEN_CONTEXT_FRAME_INDICES[0]
                    and candidate_indices[-1] == QWEN_CONTEXT_FRAME_INDICES[-1]
                    and candidate_indices == sorted(candidate_indices)
                    and len(candidate_prompts) == 3
                    and all(
                        isinstance(item, Mapping) and item.get("frame_index") == frame_index
                        for item, frame_index in zip(candidate_prompts, candidate_indices, strict=True)
                    )
                    and len({str(item.get("prompt", "")).strip() for item in candidate_prompts}) == 3
                )
            return False

        # Give Qwen up to two corrective passes when the returned JSON has a
        # malformed topology or frame mapping. The retry only restates the
        # schema contract; it never locally chooses a reference policy.
        retry_payload = payload
        valid_three_anchor_indices = ", ".join(
            str([QWEN_CONTEXT_FRAME_INDICES[0], middle, QWEN_CONTEXT_FRAME_INDICES[-1]])
            for middle in QWEN_CONTEXT_FRAME_INDICES[1:-1]
        )
        for _ in range(4):
            if valid_reference_plan(result):
                break
            retry_content = retry_payload["messages"][1]["content"]
            if not isinstance(retry_content, list):
                break
            retry_content = list(retry_content) + [{
                "type": "text",
                "text": (
                    "Correction: the previous JSON reference plan failed validation. Return exactly one of "
                    "video_only, one_anchor, or three_anchor. Every image_edit_prompts item must have a "
                    "frame_index exactly equal to its corresponding anchor_frame_indices item. For three_anchor, "
                    "return exactly three distinct prompts and distinct chronological anchors: use the first "
                    "supplied checkpoint as the first index, the last supplied checkpoint as the last index, and "
                    "one interior supplied checkpoint in the middle. Do not use an event-onset frame in place of "
                    "the first supplied checkpoint. The three prompt strings must be textually distinct and must "
                    "describe the visible target for their actual start, middle, and end source frames separately, "
                    "even when the same edited state must persist. If the raw requirement only changes speed, timing, duration, "
                    "or another temporal rate, return video_only with empty anchor and image_edit_prompts lists."
                    f" For the supplied candidate checkpoints, the only valid three_anchor index lists are: "
                    f"{valid_three_anchor_indices}. Each of the three prompts must contain at least one "
                    "frame-role-specific visible sentence that is absent from the other two prompts."
                    " Rewrite the following invalid JSON rather than repeating it: "
                    + response_text
                ),
            }]
            retry_payload = dict(payload)
            retry_payload["messages"] = [dict(retry_payload["messages"][0]), dict(retry_payload["messages"][1])]
            retry_payload["messages"][1]["content"] = retry_content
            response_text, response = self.complete(retry_payload)
            result = parse_json_object(response_text, "Qwen-VL failure repair planner")
        policy = str(result.get("reference_policy", "")).strip().lower()
        if policy not in {"video_only", "one_anchor", "three_anchor"}:
            raise ApimartError("Qwen-VL returned an invalid fallback reference_policy")
        indices = result.get("anchor_frame_indices", [])
        prompts = result.get("image_edit_prompts", [])
        if not isinstance(indices, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
            raise ApimartError("Qwen-VL fallback anchor_frame_indices must be an integer list")
        if policy == "video_only":
            if indices or prompts:
                raise ApimartError("video_only fallback must not return image anchors")
        else:
            expected = 1 if policy == "one_anchor" else 3
            if len(indices) != expected or not isinstance(prompts, list) or len(prompts) != expected:
                raise ApimartError("Qwen-VL fallback returned an incorrect anchor/prompt count")
            if len(set(indices)) != len(indices) or any(i not in QWEN_CONTEXT_FRAME_INDICES for i in indices):
                raise ApimartError("Qwen-VL fallback selected unavailable or duplicate anchor frames")
            for item, frame_index in zip(prompts, indices, strict=True):
                if not isinstance(item, Mapping) or item.get("frame_index") != frame_index:
                    raise ApimartError("Qwen-VL fallback image prompt frame mapping is invalid")
                if not str(item.get("prompt", "")).strip():
                    raise ApimartError("Qwen-VL fallback image-edit prompt is empty")
            if policy == "three_anchor" and len({str(item.get("prompt", "")).strip() for item in prompts}) != 3:
                raise ApimartError("three-anchor fallback prompts must be distinct")
            if policy == "three_anchor":
                if (
                    indices[0] != QWEN_CONTEXT_FRAME_INDICES[0]
                    or indices[-1] != QWEN_CONTEXT_FRAME_INDICES[-1]
                    or not QWEN_CONTEXT_FRAME_INDICES[0] < indices[1] < QWEN_CONTEXT_FRAME_INDICES[-1]
                ):
                    raise ApimartError(
                        "Qwen-VL three_anchor must use the first and last supplied checkpoints "
                        "with one interior checkpoint"
                    )
                # Keep the model's chronological mapping exactly as returned.
                # Never silently remap an event-specific prompt onto frame 0.
                if indices != sorted(indices):
                    raise ApimartError("Qwen-VL three_anchor indices must be chronological")
        notes = str(result.get("repair_notes", "")).strip()
        return {
            "model": self.model,
            "reference_policy": policy,
            "reference_image_count": 0 if policy == "video_only" else 1 if policy == "one_anchor" else 3,
            "anchor_frame_indices": indices,
            "image_edit_prompts": prompts,
            "repair_notes": notes,
            "usage": dict(response.get("usage", {})) if isinstance(response.get("usage"), Mapping) else {},
        }

    def observe(
        self,
        frames: Sequence[Path],
        atomic_prompt: str,
        source_frames: Sequence[Path] = (),
    ) -> dict[str, Any]:
        """Judge one atomic edit from paired source/output checkpoints."""

        observer_indices = POST_EDIT_OBSERVER_FRAME_INDICES
        if len(frames) != len(observer_indices):
            raise ApimartError(
                "Qwen-VL success gate requires exactly ten uniformly sampled observation frames: "
                f"{len(frames)} != {len(observer_indices)}"
            )
        if source_frames and len(source_frames) != len(observer_indices):
            raise ApimartError(
                "Qwen-VL source comparison requires exactly ten observation frames: "
                f"{len(source_frames)} != {len(observer_indices)}"
            )
        try:
            system = render_prompt("qwen_observer_system.txt")
            user = render_prompt("qwen_observer_user.txt", raw_prompt=atomic_prompt)
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        # Keep the system contract in the system message only. Repeating it
        # inside the multimodal user content wastes context and can make the
        # judge over-weight generic rules over the actual video evidence.
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        if source_frames:
            for frame_index, frame in zip(observer_indices, source_frames, strict=True):
                content.append({"type": "text", "text": f"Source frame {frame_index}:"})
                content.append({"type": "image_url", "image_url": {"url": self.image_data_url(frame)}})
        for frame_index, frame in zip(observer_indices, frames, strict=True):
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


def observe_stage_output(
    refiner: DashScopeVisionRefiner,
    output: Path,
    stage_dir: Path,
    task_id: str,
    stage_label: str,
    atomic_prompt: str,
    source_video: Path | None = None,
    prompt_provenance: str = "runner_stage_atomic_prompt",
) -> dict[str, Any]:
    """Extract paired source/output checkpoints and persist the success gate."""

    observation_dir = stage_dir / "observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"task_{task_id}_{stage_label}_output_observation_frame"
    frames: list[Path] = []
    source_frames: list[Path] = []
    if source_video is not None:
        for frame_index in POST_EDIT_OBSERVER_FRAME_INDICES:
            source_frame = observation_dir / f"{file_prefix.replace('_output_', '_source_')}_{frame_index:03d}.png"
            select_keyframe(source_video, source_frame, frame_index)
            source_frames.append(source_frame)
    for frame_index in POST_EDIT_OBSERVER_FRAME_INDICES:
        frame = observation_dir / f"{file_prefix}_{frame_index:03d}.png"
        select_keyframe(output, frame, frame_index)
        frames.append(frame)
    try:
        result = validate_observation(refiner.observe(frames, atomic_prompt, source_frames))
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
        "kind": "qwen_vl_uniform_temporal_success_gate_v2",
        "stage": stage_label,
        "task_id": task_id,
        "frame_indices": list(POST_EDIT_OBSERVER_FRAME_INDICES),
        "frames": [str(frame) for frame in frames],
        "source_frames": [str(frame) for frame in source_frames],
        "atomic_prompt": atomic_prompt,
        "prompt_provenance": prompt_provenance,
        **result,
    }
    write_json(observation_dir / "observation.json", record)
    print(json.dumps({
        "event": "post_edit_observation",
        "stage": stage_label,
        "success": result.get("success"),
        "confidence": result.get("confidence"),
        "observation": result.get("observation", ""),
        "frame_indices": list(POST_EDIT_OBSERVER_FRAME_INDICES),
    }, ensure_ascii=False), flush=True)
    return record
