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
    "qwen_h3_validation_retry.txt",
    "qwen_observer_system.txt",
    "qwen_observer_user.txt",
    "h3_temporal_anchor.txt",
    "h3_one_anchor_repair.txt",
    "h3_video_only_repair.txt",
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
