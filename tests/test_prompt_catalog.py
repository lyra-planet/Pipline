from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from apimart_h3_pipeline.prompt_catalog import (  # noqa: E402
    PromptResourceError,
    load_prompt,
    load_repair_clauses,
    render_prompt,
)


PROMPT_FILES = (
    "qwen_reference_system.txt",
    "qwen_reference_global_style_suffix.txt",
    "qwen_reference_user.txt",
    "qwen_parent_frame_label.txt",
    "qwen_picture_label.txt",
    "qwen_picture_role_label.txt",
    "qwen_h3_reference_contract.txt",
    "qwen_h3_no_reference_contract.txt",
    "qwen_h3_system.txt",
    "qwen_h3_global_style_suffix.txt",
    "qwen_h3_failure_evidence.txt",
    "qwen_h3_user.txt",
    "qwen_observer_system.txt",
    "qwen_observer_user.txt",
    "h3_temporal_anchor.txt",
    "h3_one_anchor_repair.txt",
    "h3_video_only_repair.txt",
    "h3_dynamic_action_contract.txt",
    "h3_camera_motion_contract.txt",
    "h3_camera_motion_clauses.json",
    "image_edit_preservation.txt",
    "repair_clauses.json",
)


def test_all_prompt_resources_are_readable_without_cwd_dependency(tmp_path: Path) -> None:
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        for name in PROMPT_FILES:
            assert load_prompt(name).strip(), name
    finally:
        os.chdir(old_cwd)


def test_prompt_resource_names_reject_path_traversal() -> None:
    for name in ("../secret.txt", "prompts/x.txt", "/tmp/x.txt", "", "."):
        with pytest.raises(PromptResourceError):
            load_prompt(name)


def test_render_prompt_requires_all_template_values() -> None:
    with pytest.raises(PromptResourceError, match="missing value"):
        render_prompt("qwen_h3_user.txt")

    rendered = render_prompt("h3_temporal_anchor.txt", requirement="Change color.", anchor_lines="anchors")
    assert "${" not in rendered
    assert "Change color." in rendered
    assert "anchors" in rendered


def test_qwen_h3_prompt_contract_puts_operation_before_references() -> None:
    system = load_prompt("qwen_h3_system.txt")
    user = render_prompt("qwen_h3_user.txt", raw_prompt="Move the camera slowly to the right.")
    contract = render_prompt(
        "qwen_h3_reference_contract.txt",
        picture_tags="<Picture 1>",
        role_contract="<Picture 1> = edited primary anchor, source frame 0",
    )
    assert "Begin h3_prompt with the exact raw atomic requirement text verbatim" in system
    assert user.index("Start h3_prompt with the raw atomic edit requirement") < user.index("Only after")
    assert contract.index("first sentence must state the raw operation") < contract.index("assign appearance")


def test_three_anchor_resource_contains_the_frame_lock_contract() -> None:
    rendered = render_prompt(
        "qwen_h3_reference_contract.txt",
        picture_tags="<Picture 1>, <Picture 2>, <Picture 3>",
        role_contract=(
            "<Picture 1> = edited start anchor, source frame 0\n"
            "<Picture 2> = edited primary anchor, source frame 53\n"
            "<Picture 3> = edited end anchor, source frame 106"
        ),
    )
    assert all(tag in rendered for tag in ("<Picture 1>", "<Picture 2>", "<Picture 3>", "<Video 1>"))
    assert "source frame 0" in rendered
    assert "source frame 53" in rendered
    assert "source frame 106" in rendered
    assert "first sentence must state the raw operation" in rendered


def test_repair_clause_actions_are_a_closed_nonempty_mapping() -> None:
    clauses = load_repair_clauses()
    expected = {
        "strengthen_edit",
        "strengthen_identity_preservation",
        "strengthen_previous_stage_preservation",
        "use_three_anchor",
        "strengthen_motion",
        "strengthen_composition",
    }
    assert set(clauses) == expected
    assert all(value.strip() for value in clauses.values())


def test_image_edit_prompt_adds_only_the_shared_preservation_constraint() -> None:
    from apimart_h3_pipeline.prompt_catalog import image_edit_prompt

    raw = "Transform the background to a dimly lit room."
    rendered = image_edit_prompt(raw)
    assert rendered.startswith(raw)
    assert rendered.endswith("Preserve all other elements exactly as they are.")
    assert image_edit_prompt(rendered) == rendered


def test_reference_planner_contract_matches_image_editor_prompt_policy() -> None:
    rendered = load_prompt("qwen_reference_system.txt")
    assert "raw atomic requirement plus the shared instruction" in rendered
    assert "raw atomic requirement verbatim" not in rendered


def test_camera_motion_contract_is_explicit_about_pan_vs_zoom() -> None:
    rendered = render_prompt("h3_camera_motion_contract.txt")
    assert "requested camera movement" in rendered
    assert "object motion" in rendered


def test_dynamic_action_contract_is_temporal_and_idempotent() -> None:
    from apimart_h3_pipeline.prompt_catalog import dynamic_action_prompt

    raw = "Change the girl's action to a high vertical jump with arms outstretched."
    rendered = dynamic_action_prompt(raw)
    assert rendered.startswith(raw)
    assert "natural onset and progression" in rendered
    assert dynamic_action_prompt(rendered) == rendered
