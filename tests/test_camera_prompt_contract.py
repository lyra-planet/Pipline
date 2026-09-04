from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from apimart_h3_pipeline.bridge.helpers import deterministic_repair_h3_prompt
from apimart_h3_pipeline.core.policy import camera_motion_kind, expected_reference_roles, is_camera_motion_edit, is_dynamic_action_edit, reference_policy
from apimart_h3_pipeline.core.repair_policy import FailureDiagnosisAndRepair
from apimart_h3_pipeline.prompt_catalog import camera_motion_prompt, dynamic_action_prompt
from apimart_h3_pipeline.providers.vision_refiner import DashScopeVisionRefiner


def context_frames(root: Path) -> list[Path]:
    frames: list[Path] = []
    for index in (0, 26, 53, 80, 106):
        frame = root / f"frame_{index:03d}.png"
        Image.new("RGB", (8, 8), (index % 255, 20, 30)).save(frame)
        frames.append(frame)
    return frames


def refiner_with_response(response_prompt: str) -> DashScopeVisionRefiner:
    refiner = object.__new__(DashScopeVisionRefiner)
    refiner.model = "qwen-vl-plus-test"
    refiner.complete = lambda _payload: (
        json.dumps({"h3_prompt": response_prompt, "frame_observation": "observed"}),
        {"usage": {}},
    )
    return refiner


def test_camera_prompt_gets_explicit_physical_motion_contract(tmp_path: Path) -> None:
    raw = "Move the camera slowly to the right."
    response_prompt = f"{raw} Use a clear, strong horizontal camera pan to the right while keeping the subject in focus."
    refiner = refiner_with_response(response_prompt)

    result = refiner.compose_h3_prompt(
        context_frames(tmp_path), [], raw, False,
    )

    assert result["h3_prompt"] == response_prompt
    assert "strong horizontal camera pan" in result["h3_prompt"]
    assert "camera pan" in result["h3_prompt"]


def test_non_camera_video_only_prompt_has_no_camera_contract(tmp_path: Path) -> None:
    raw = "Add a soft wind sound."
    refiner = refiner_with_response("Apply only this edit to <Video 1>: sound")

    result = refiner.compose_h3_prompt(
        context_frames(tmp_path), [], raw, False,
    )

    assert result["h3_prompt"] == "Apply only this edit to <Video 1>: sound"


def test_pure_action_policy_is_video_only_and_contract_preserves_onset() -> None:
    raw = "Change the girl's action to a high vertical jump with arms outstretched."
    assert is_dynamic_action_edit(raw)
    policy = reference_policy(raw)
    assert policy["needs_reference_image"] is False
    assert policy["reference_image_count"] == 0
    assert dynamic_action_prompt(raw).startswith(raw)


def test_pure_action_h3_prompt_uses_video_only_temporal_contract(tmp_path: Path) -> None:
    raw = "Change the girl's action to a high vertical jump with arms outstretched."
    refiner = refiner_with_response("Apply only this edit to <Video 1>: jump")
    result = refiner.compose_h3_prompt(context_frames(tmp_path), [], raw, False)
    assert result["picture_count"] == 0
    assert result["h3_prompt"] == "Apply only this edit to <Video 1>: jump"


def test_mixed_action_object_edit_is_not_video_only() -> None:
    assert not is_dynamic_action_edit(
        "Replace the selfie-taking action with both subjects reading a broadsheet newspaper together."
    )
    assert not is_dynamic_action_edit("Change the action to writing a visible digit on a whiteboard.")
    assert not is_dynamic_action_edit("Change the person's action to touching a green leaf with their hand.")
    assert not is_dynamic_action_edit("Change the person's position to stand on the right side of the frame.")
    assert not is_dynamic_action_edit("Change the climber's action to perform a dynamic jump towards a higher handhold.")


def test_dynamic_action_retry_keeps_video_only_temporal_contract() -> None:
    raw = "Change the girl's action to a high vertical jump with arms outstretched."
    policy = FailureDiagnosisAndRepair()
    diagnosis = policy.diagnose(
        {"success": False, "failure_type": "edit_missing", "observation": "jump absent", "confidence": 0.95},
        stage_id="S1",
        current_requirement=raw,
        failed_prompt=f"Apply only this edit to <Video 1>: {raw}",
    )
    repair = policy.repair(
        diagnosis,
        stage_id="S1",
        current_requirement=raw,
        failed_prompt=f"Apply only this edit to <Video 1>: {raw}",
        retry_index=1,
        original_policy=reference_policy(raw),
    )
    repaired = deterministic_repair_h3_prompt(raw, 0, (), repair)
    assert "<Picture" not in repaired
    assert "natural onset and progression" in repaired


def test_camera_contract_is_enforced_when_qwen_attaches_a_reference(tmp_path: Path) -> None:
    raw = "Move the camera slowly to the right."
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(reference)
    refiner = refiner_with_response(
        f"{raw} Use <Picture 1> as the visual appearance target and preserve <Video 1> for the camera motion."
    )

    result = refiner.compose_h3_prompt(
        context_frames(tmp_path), [reference], raw, False, expected_reference_roles(1),
    )

    assert result["h3_prompt"].startswith(raw)
    assert "<Picture 1>" in result["h3_prompt"]
    assert "<Video 1>" in result["h3_prompt"]


def test_camera_classifier_is_narrow_to_explicit_camera_terms() -> None:
    assert is_camera_motion_edit("Move the camera to the left.")
    assert is_camera_motion_edit("Add a gentle push-in.")
    assert not is_camera_motion_edit("Move the person to the left side of the frame.")
    assert not is_camera_motion_edit("Add a digital camera to the table.")
    assert not is_camera_motion_edit("Change the camera battery.")
    assert camera_motion_kind("Move the camera to the right.") == "pan"
    assert camera_motion_kind("Add a gentle camera pull-back.") == "pull_out"
    assert camera_motion_kind("Tilt the camera upward.") == "tilt"
    assert camera_motion_kind("Track the camera across the street.") == "tracking"
    assert camera_motion_kind("Pan across the scene.") == "pan"


def test_camera_motion_prompt_is_idempotent_and_keeps_the_atomic_requirement() -> None:
    raw = "Move the camera to the left."
    rendered = camera_motion_prompt(raw)
    assert rendered.startswith(raw)
    assert "smooth horizontal pan" in rendered
    assert camera_motion_prompt(rendered) == rendered


def test_camera_motion_prompt_adds_only_the_named_motion_clarification() -> None:
    raw = "Move the camera slowly to the right."
    rendered = camera_motion_prompt(raw)
    assert rendered.startswith(raw)
    assert "smooth horizontal pan" in rendered
    assert "push-in" not in rendered
    assert "pull-out" not in rendered
    assert "zoom" not in rendered
    assert "subject motion" not in rendered


def test_non_camera_content_edit_is_not_rewritten() -> None:
    raw = "Add a digital camera to the table."
    assert camera_motion_prompt(raw) == raw


def test_deterministic_camera_retry_keeps_the_motion_clarification() -> None:
    raw = "Move the camera slowly to the right."
    policy = FailureDiagnosisAndRepair()
    diagnosis = policy.diagnose(
        {"success": False, "failure_type": "edit_missing", "observation": "camera move absent", "confidence": 0.95},
        stage_id="S2",
        current_requirement=raw,
        failed_prompt=f"Apply only this edit to <Video 1>: {raw}",
    )
    repair = policy.repair(
        diagnosis,
        stage_id="S2",
        current_requirement=raw,
        failed_prompt=f"Apply only this edit to <Video 1>: {raw}",
        retry_index=1,
        original_policy={"needs_reference_image": True, "reference_image_count": 1},
    )
    repaired = deterministic_repair_h3_prompt(raw, 1, expected_reference_roles(1), repair)
    assert "smooth horizontal pan" in repaired
    assert "<Video 1>" in repaired


def test_qwen_prompt_is_used_without_reference_contract_validation(tmp_path: Path) -> None:
    raw = "Transform the background into a kitchen."
    response_prompt = "Use the attached image as a guide; keep the person unchanged."
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(reference)
    calls = 0
    refiner = object.__new__(DashScopeVisionRefiner)
    refiner.model = "qwen-vl-plus-test"

    def complete(_payload):
        nonlocal calls
        calls += 1
        return json.dumps({"h3_prompt": response_prompt, "frame_observation": "observed"}), {"usage": {}}

    refiner.complete = complete
    result = refiner.compose_h3_prompt(
        context_frames(tmp_path), [reference], raw, False,
    )

    assert result["h3_prompt"] == response_prompt
    assert result["repair_attempts"] == []
    assert calls == 1


def test_non_json_qwen_prompt_is_passed_through_without_rejection(tmp_path: Path) -> None:
    response_prompt = "Edit the background only and keep everything else unchanged."
    refiner = object.__new__(DashScopeVisionRefiner)
    refiner.model = "qwen-vl-plus-test"
    refiner.complete = lambda _payload: (response_prompt, {"usage": {}})

    result = refiner.compose_h3_prompt(
        context_frames(tmp_path), [], "Change the background.", False,
    )

    assert result["h3_prompt"] == response_prompt
    assert result["repair_attempts"] == []
