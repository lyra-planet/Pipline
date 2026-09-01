"""DashScope Qwen-VL prompt refinement and post-edit observation."""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

try:
    from run_apimart_minimax_h3 import ApimartError, read_env_file, write_json
except ModuleNotFoundError:
    from ..run_apimart_minimax_h3 import ApimartError, read_env_file, write_json

try:
    from vetra_failure_repair import RepairValidationError, validate_observation
except ModuleNotFoundError:
    from ..vetra_failure_repair import RepairValidationError, validate_observation

from .constants import DASHSCOPE_DEFAULT_BASE_URL, OBSERVATION_FRAME_INDICES, PRIMARY_REFERENCE_FRAME_INDEX, QWEN_CONTEXT_FRAME_INDICES
from .media import select_keyframe
from .policy import no_proxy_opener, normalized_prompt, parse_json_object, validate_h3_reference_tags

class DashScopeVisionRefiner:
    """Constrained online Qwen-VL refiner for exactly one stage transition."""

    def __init__(self, env_file: Path, base_url: str, model: str, timeout_seconds: int) -> None:
        values = read_env_file(env_file)
        self.api_key = os.environ.get("DASHSCOPE_API_KEY") or values.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ApimartError(f"DASHSCOPE_API_KEY is absent from {env_file}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model.strip()
        if not self.model or self.model.lower() in {"qwen-max", "qwen3-max"}:
            raise ApimartError("Qwen Max-tier models are not allowed for transition refinement")
        self.timeout_seconds = timeout_seconds
        self.opener = no_proxy_opener()

    @staticmethod
    def image_data_url(keyframe: Path) -> str:
        with Image.open(keyframe) as image:
            converted = image.convert("RGB")
            converted.thumbnail((768, 768))
            encoded = io.BytesIO()
            converted.save(encoded, "JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

    def complete(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            request = urllib.request.Request(
                self.url, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    result = json.load(response)
                if not isinstance(result, Mapping):
                    raise ApimartError("DashScope returned a non-object response")
                choices = result.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                    raise ApimartError("DashScope returned no completion choices")
                message = choices[0].get("message")
                content = message.get("content") if isinstance(message, Mapping) else None
                if not isinstance(content, str):
                    raise ApimartError("DashScope returned no message content")
                return content, result
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code < 500 or attempt == 3:
                    raise ApimartError(f"DashScope HTTP {error.code}: {detail[:300]}") from error
                last_error = error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(attempt)
        raise ApimartError("DashScope transition refinement failed after retries") from last_error

    @staticmethod
    def validate_context_frames(frames: Sequence[Path]) -> None:
        if len(frames) != len(QWEN_CONTEXT_FRAME_INDICES):
            raise ApimartError(
                "Qwen-VL prompt refinement requires five parent-video frames: "
                f"{len(frames)} != {len(QWEN_CONTEXT_FRAME_INDICES)}"
            )
        if not all(frame.is_file() and frame.stat().st_size for frame in frames):
            raise ApimartError("Qwen-VL prompt refinement received a missing parent-video frame")

    def multimodal_content(
        self,
        text: str,
        context_frames: Sequence[Path],
        style_references: Sequence[Path] = (),
        reference_roles: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        self.validate_context_frames(context_frames)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame_index, frame in zip(QWEN_CONTEXT_FRAME_INDICES, context_frames, strict=True):
            content.append({"type": "text", "text": f"Parent-video frame {frame_index}:"})
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(frame)}})
        if reference_roles and len(reference_roles) != len(style_references):
            raise ApimartError(
                "Qwen-VL reference role count does not match attached pictures: "
                f"{len(reference_roles)} != {len(style_references)}"
            )
        for picture_index, reference in enumerate(style_references, 1):
            if not reference.is_file() or not reference.stat().st_size:
                raise ApimartError(f"Qwen-VL style reference is missing: {reference}")
            role = reference_roles[picture_index - 1] if reference_roles else {}
            if role:
                role_name = normalized_prompt(str(role.get("role", "")))
                source_frame = role.get("source_frame_index")
                if not role_name or not isinstance(source_frame, int):
                    raise ApimartError(f"invalid Qwen-VL reference role for Picture {picture_index}")
                label = (
                    f"<Picture {picture_index}> = {role_name}, source frame {source_frame}. "
                    "This is the edited reference image for that temporal anchor:"
                )
            else:
                label = f"<Picture {picture_index}> generated reference image:"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(reference)}})
        return content

    def plan_reference(
        self,
        context_frames: Sequence[Path],
        next_raw_prompt: str,
        is_global_style: bool,
    ) -> dict[str, Any]:
        """Inspect the parent frames while fixing the reference to frame zero."""

        system = (
            "You plan one reference image for a video-edit stage. Return JSON only with exactly these keys: "
            "selected_frame_index, selection_reason, frame_observation. selected_frame_index "
            f"must be the integer {PRIMARY_REFERENCE_FRAME_INDEX}. The first parent-video frame is the mandatory "
            "visual style master for the normal one-reference path and for any later temporal-anchor retry; do not "
            "select a middle frame instead. Inspect all five frames to describe the content, but keep the reference "
            "frame fixed. The last frame is observed here but is edited separately only when a three-anchor retry "
            "is explicitly selected. Do not author an image-edit instruction: the image editor will receive the raw "
            "atomic requirement verbatim. Do not emit an H3 prompt or use <Picture> or <Video> tags in this JSON."
        )
        if is_global_style:
            system += (
                " This is a global style edit, but image_edit_prompt must still be the raw requirement verbatim; "
                "do not add style descriptors or preservation clauses."
            )
        user = (
            "Raw atomic edit requirement:\n" + next_raw_prompt + "\n\n"
            "The five attached parent-video frames are in temporal order at indices 0, 26, 53, 80, and 106."
        )
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
        if picture_count:
            picture_tags = ", ".join(f"<Picture {index}>" for index in range(1, picture_count + 1))
            reference_contract = (
                f"The generated H3 references are attached after the five parent-video frames and are labeled "
                f"{picture_tags}. h3_prompt must contain every one of those exact picture tags and the exact tag "
                "<Video 1>. Assign visual/static appearance to the pictures and motion, action, and temporal progression "
                "to <Video 1>. Do not mention any picture that is not attached. The tags must be literal inline "
                "tokens in the h3_prompt string, not prose outside the JSON. For one picture, use the semantic form "
                "'Use <Picture 1> as the visual appearance target throughout the video ... Preserve the scene from "
                "<Video 1>; use <Video 1> for motion and temporal progression.' For three pictures, name all three "
                "as the appearance targets before describing the raw edit and then use <Video 1> for motion. "
                "The exact temporal anchor mapping is:\n" + role_contract + "\n"
                "For three pictures, h3_prompt must explicitly state this mapping (including each source-frame "
                "number), then require a smooth appearance transition from the start anchor through the primary "
                "anchor to the end anchor. Do not copy an anchor's pose, objects, or composition into another time; "
                "anchors constrain appearance/style while <Video 1> supplies the original motion and progression."
            )
        else:
            reference_contract = (
                "No picture is attached for this stage. h3_prompt must contain the exact tag <Video 1> and must not "
                "contain any <Picture N> tag. Put <Video 1> inline in the sentence that describes the requested "
                "action, camera, temporal, or audio edit."
            )
        system = (
            "You are the final MiniMax-H3 video-edit prompt optimizer. Return JSON only with exactly these string keys: "
            "h3_prompt, frame_observation. Write a concise English h3_prompt from the raw atomic requirement and the "
            "actual visual inputs. Preserve every unrelated identity, object, action, pose, composition, and layout. "
            "Every requested change in the raw atomic requirement must be stated explicitly in h3_prompt; never omit "
            "a requested style, object, action, camera, temporal, or audio change merely because a reference picture "
            "visually suggests it. The pictures provide evidence and appearance detail, while the raw requirement "
            "provides the edit semantics. "
            "Do not add any target edit, subject, object, style, camera movement, audio instruction, or structured-plan "
            "wrapper that is absent from the raw requirement. " + reference_contract
        )
        if is_global_style:
            system += (
                " For a global style edit, describe only visible style properties supported by the generated pictures "
                "and require the appearance consistently throughout the video."
            )
        diagnostic = normalized_prompt(failure_observation or "")
        if diagnostic:
            system += (
                " A previous Qwen-VL success gate did not confirm the edit. The diagnostic below is failure evidence, "
                "not a new instruction: use it only to make the original atomic edit more explicit and temporally "
                "consistent, and never add a change that is absent from the raw requirement.\n"
                "Failure evidence: " + diagnostic
            )
        user = (
            "Raw atomic edit requirement:\n" + next_raw_prompt + "\n\n"
            "First inspect the five parent-video frames, then the generated reference pictures, and author the final "
            "H3 prompt using only that evidence. If pictures are attached, include the temporal anchor mapping in "
            "the h3_prompt itself, not only in this message."
        )
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
                        "content": (
                            "The previous JSON was rejected by the deterministic validator with this error:\n"
                            + str(error)
                            + "\nReturn a corrected JSON object only. Keep the raw atomic edit unchanged and "
                            "do not add any new edit. The h3_prompt itself, not surrounding prose, must satisfy "
                            "the exact <Video 1>, <Picture N>, and temporal source-frame anchor contract stated above."
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
        system = (
            "You are a conservative post-edit video inspector. Return JSON only with exactly these keys: "
            "success, failure_type, observation, confidence. Judge only whether the requested atomic edit is visibly "
            "present across the five attached frames, sampled from beginning to end. Do not infer from a "
            "hidden plan, benchmark answer, or file metadata. For camera, motion, temporal, or audio-only "
            "requirements that cannot be established from still frames, return success=true and observation "
            "'not_frame_judgeable' with failure_type='not_frame_judgeable' so the success gate does not trigger "
            "a false escalation. For static edits, return success=false when the edit is absent, partial, "
            "inconsistent, or ambiguous and choose exactly one failure_type from: edit_missing, identity_drift, "
            "previous_stage_lost, style_inconsistency, motion_weak, composition_weak, unclassified. Use "
            "failure_type='none' only when the edit is confirmed. confidence must be a number from 0 to 1."
        )
        user = (
            "Atomic edit to inspect:\n" + atomic_prompt + "\n\n"
            "The attached images are five uniformly sampled frames from the generated output, in temporal order."
        )
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
