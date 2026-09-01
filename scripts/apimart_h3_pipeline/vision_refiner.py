"""DashScope Qwen-VL prompt refinement and post-edit observation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from run_apimart_minimax_h3 import ApimartError, write_json
except ModuleNotFoundError:
    from ..run_apimart_minimax_h3 import ApimartError, write_json

try:
    from vetra_failure_repair import RepairValidationError, validate_observation
except ModuleNotFoundError:
    from ..vetra_failure_repair import RepairValidationError, validate_observation

from .constants import OBSERVATION_FRAME_INDICES, PRIMARY_REFERENCE_FRAME_INDEX, QWEN_CONTEXT_FRAME_INDICES
from .media import select_keyframe
from .policy import normalized_prompt, parse_json_object, validate_h3_reference_tags
from .prompt_catalog import PromptResourceError, render_prompt
from .vision_client import DashScopeClient

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
            "max_tokens": 520,
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
        # changes content that the atomic request never asked to change.  Qwen
        # still selects the frame and records its proposed wording for audit,
        # but the editor receives the original compiled prompt verbatim.
        raw_image_edit_prompt = next_raw_prompt.strip()
        if not raw_image_edit_prompt:
            raise ApimartError("raw atomic prompt is empty for image editing")
        return {
            "model": self.model,
            "selected_frame_index": selected_frame_index,
            "selection_reason": normalized_prompt(str(result.get("selection_reason", ""))),
            "image_edit_prompt": raw_image_edit_prompt,
            "image_edit_prompt_source": "raw_atomic_prompt_verbatim",
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
        if reference_roles and len(reference_roles) != picture_count:
            raise ApimartError(
                "final Qwen prompt requires one temporal role per attached picture: "
                f"{len(reference_roles)} != {picture_count}"
            )
        role_lines = [
            f"<Picture {index}> = {str(role.get('role', '')).strip()}, "
            f"source frame {int(role.get('source_frame_index'))}"
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
            "max_tokens": 520,
        }
        rejected_outputs: list[dict[str, str]] = []
        response: Mapping[str, Any] = {}
        for repair_attempt in range(3):
            response_text, response = self.complete(payload)
            try:
                result = parse_json_object(response_text, "Qwen-VL final H3 prompt")
                h3_prompt = validate_h3_reference_tags(
                    str(result.get("h3_prompt", "")), picture_count, reference_roles,
                )
            except ApimartError as error:
                rejected_outputs.append({"output": response_text, "validation_error": str(error)})
                if repair_attempt == 2:
                    raise ApimartError(
                        "Qwen-VL could not produce a reference-consistent H3 prompt after 3 attempts"
                    ) from error
                payload = dict(payload)
                payload["messages"] = [
                    *payload["messages"],
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": render_prompt(
                            "qwen_h3_validation_retry.txt",
                            validation_error=str(error),
                        ),
                    },
                ]
                continue
            usage = response.get("usage")
            if picture_count == 0:
                # A video-only atomic edit must not acquire new scene
                # semantics from Qwen's frame caption (for example, naming
                # objects or materials that were absent from the raw request).
                # Keep the required <Video 1> binding, but pass the original
                # camera/action/timing instruction verbatim to H3.
                h3_prompt = f"Apply only this edit to <Video 1>: {next_raw_prompt.strip()}"
            return {
                "model": self.model,
                "h3_prompt": h3_prompt,
                "h3_prompt_source": "raw_atomic_prompt_with_video_binding"
                if picture_count == 0 else "qwen_reference_contract",
                "frame_observation": normalized_prompt(str(result.get("frame_observation", ""))),
                "picture_count": picture_count,
                "is_global_style": is_global_style,
                "repair_attempts": rejected_outputs,
                "usage": dict(usage) if isinstance(usage, Mapping) else {},
            }
        raise ApimartError("Qwen-VL H3 prompt repair loop exited unexpectedly")

    def observe(self, frames: Sequence[Path], atomic_prompt: str) -> dict[str, Any]:
        """Judge one atomic edit from five uniformly sampled output frames."""

        if len(frames) != len(OBSERVATION_FRAME_INDICES):
            raise ApimartError(
                "Qwen-VL success gate requires exactly five observation frames: "
                f"{len(frames)} != {len(OBSERVATION_FRAME_INDICES)}"
            )
        try:
            system = render_prompt("qwen_observer_system.txt")
            user = render_prompt("qwen_observer_user.txt", raw_prompt=atomic_prompt)
        except PromptResourceError as error:
            raise ApimartError(str(error)) from error
        content: list[dict[str, Any]] = [{"type": "text", "text": system + "\n\n" + user}]
        content.extend(
            {"type": "image_url", "image_url": {"url": self.image_data_url(frame)}}
            for frame in frames
        )
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
) -> dict[str, Any]:
    """Extract five temporal checkpoints and persist the Qwen-VL success gate."""

    observation_dir = stage_dir / "observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"task_{task_id}_{stage_label}_output_observation_frame"
    frames: list[Path] = []
    for frame_index in OBSERVATION_FRAME_INDICES:
        frame = observation_dir / f"{file_prefix}_{frame_index:03d}.png"
        select_keyframe(output, frame, frame_index)
        frames.append(frame)
    try:
        result = validate_observation(refiner.observe(frames, atomic_prompt))
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
