#!/usr/bin/env python3
"""Generate GRSAI references from saved fallback plans, without MiniMax-H3."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.core.constants import QWEN_CONTEXT_FRAME_INDICES  # noqa: E402
from apimart_h3_pipeline.media import select_keyframe  # noqa: E402
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.grsai import GrsaiImageEditor  # noqa: E402
from apimart_h3_pipeline.providers.vision_refiner import DashScopeVisionRefiner  # noqa: E402


DEFAULT_PLAN_RUN = Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_50_prompt_only_20260908_v2")
DEFAULT_OUT_DIR = Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_50_references_20260908")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-run", type=Path, default=DEFAULT_PLAN_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--grsai-env", type=Path, default=Path("/root/.grsai.env"))
    parser.add_argument("--task-list", type=Path)
    parser.add_argument("--count", type=int, default=50)
    return parser.parse_args()


def task_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"task_(\d+)", path.name)
    return int(match.group(1)) if match else None


def selected_tasks(args: argparse.Namespace) -> list[int]:
    if args.task_list:
        text = args.task_list.read_text(encoding="utf-8")
        # Older launcher output escaped newlines into one physical line. Accept
        # both that form and the normal one-ID-per-line format.
        text = text.replace("\\n", "\n")
        values: list[int] = []
        for line in text.splitlines():
            value = line.strip()
            if value and value.isdigit():
                values.append(int(value))
        return values[: args.count]
    # The prompt-only summary can be a partial batch summary. The durable
    # per-task fallback files are the authoritative set for this stage.
    discovered = []
    result_paths = list(args.plan_run.glob("task_*/stages/S1/fallback_prompt_only.json"))
    result_paths.sort(key=lambda item: task_id_from_path(item.parents[2]) or 10**9)
    for result_path in result_paths:
        task_id = task_id_from_path(result_path.parents[2])
        if task_id is not None:
            existing_record = result_path.parent / "reference_generation.json"
            if existing_record.is_file():
                try:
                    existing = json.loads(existing_record.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    existing = {}
                if isinstance(existing, Mapping) and existing.get("status") in {
                    "references_ready_before_minmax",
                    "video_only_no_reference_required",
                }:
                    continue
            discovered.append(task_id)
    if discovered:
        return discovered[: args.count]
    selection = args.plan_run / "selected_tasks.json"
    if selection.is_file():
        value = json.loads(selection.read_text(encoding="utf-8"))
        tasks = value.get("tasks") if isinstance(value, Mapping) else None
        if isinstance(tasks, list):
            selected = [int(item) for item in tasks if str(item).isdigit()]
            if selected:
                return selected[: args.count]
    summary = json.loads((args.plan_run / "summary.json").read_text(encoding="utf-8"))
    return [
        int(item["task_id"])
        for item in summary.get("tasks", [])
        if isinstance(item, Mapping) and str(item.get("task_id", "")).isdigit()
    ][: args.count]


def source_video_from_plan(plan: Mapping[str, Any]) -> Path:
    source_video = Path(str(plan.get("source_video", "")))
    if not source_video.is_file():
        source_task = Path(str(plan.get("source_task", "")))
        task_name = source_task.name
        source_video = source_task / "media" / f"{task_name}_initial.mp4"
    if not source_video.is_file():
        raise ApimartError(f"source video is missing: {source_video}")
    return source_video


def load_plan(plan_run: Path, task_id: int) -> tuple[dict[str, Any], Path]:
    result_path = plan_run / f"task_{task_id}" / "stages" / "S1" / "fallback_prompt_only.json"
    if not result_path.is_file():
        raise ApimartError(f"missing prompt-only result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, Mapping):
        raise ApimartError(f"prompt-only result is not an object: {result_path}")
    if result.get("status") == "fallback_error":
        raise ApimartError(f"task {task_id} has a failed fallback plan")
    plan_path = Path(str(result.get("repair_plan_path", "")))
    if not plan_path.is_file():
        plan_path = plan_run / f"task_{task_id}" / "stages" / "S1" / "bridge_for_next" / f"task_{task_id}_S1_qwen_failure_repair_plan.json"
    if not plan_path.is_file():
        raise ApimartError(f"missing Qwen repair plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or not isinstance(plan.get("result"), Mapping):
        raise ApimartError(f"invalid Qwen repair plan: {plan_path}")
    return dict(plan), result_path


def context_frame(plan_run: Path, task_id: int, source_video: Path, frame_index: int) -> Path:
    frame = (
        plan_run
        / f"task_{task_id}"
        / "stages"
        / "S1"
        / "bridge_for_next"
        / f"task_{task_id}_S1_context_frame_{frame_index:03d}.png"
    )
    if not frame.is_file() or not frame.stat().st_size:
        select_keyframe(source_video, frame, frame_index)
    return frame


def compose_reference_aware_prompt(
    refiner: DashScopeVisionRefiner,
    plan_run: Path,
    task_id: int,
    plan: Mapping[str, Any],
    references: list[str],
    result_path: Path,
) -> dict[str, Any]:
    """Compose the final H3 prompt only after the actual references exist."""

    bridge_dir = plan_run / f"task_{task_id}" / "stages" / "S1" / "bridge_for_next"
    context_frames = [
        bridge_dir / f"task_{task_id}_S1_context_frame_{frame_index:03d}.png"
        for frame_index in QWEN_CONTEXT_FRAME_INDICES
    ]
    if not all(item.is_file() and item.stat().st_size for item in context_frames):
        raise ApimartError(f"task {task_id} is missing Qwen source context frames")
    qwen_result = plan.get("result")
    if not isinstance(qwen_result, Mapping):
        raise ApimartError(f"task {task_id} has no Qwen repair result")
    indices = qwen_result.get("anchor_frame_indices")
    if len(references) == 1:
        if not isinstance(indices, list) or len(indices) != 1:
            raise ApimartError(f"task {task_id} one-anchor mapping is invalid")
        roles = [{"picture_index": 1, "role": "edited primary anchor", "source_frame_index": int(indices[0])}]
    elif len(references) == 3:
        if not isinstance(indices, list) or len(indices) != 3:
            raise ApimartError(f"task {task_id} three-anchor mapping is invalid")
        roles = [
            {"picture_index": 1, "role": "edited start anchor", "source_frame_index": int(indices[0])},
            {"picture_index": 2, "role": "edited middle anchor", "source_frame_index": int(indices[1])},
            {"picture_index": 3, "role": "edited end anchor", "source_frame_index": int(indices[2])},
        ]
    else:
        raise ApimartError(f"task {task_id} has unsupported generated reference count: {len(references)}")
    raw_prompt = str(plan.get("raw_prompt", "")).strip()
    failed_prompt = str(plan.get("failed_h3_prompt", "")).strip()
    failure_type = str(plan.get("failure_type", "unclassified")).strip()
    failure_observation = str(plan.get("failure_observation", "")).strip()
    composed = refiner.compose_h3_prompt(
        context_frames,
        [Path(item) for item in references],
        raw_prompt,
        False,
        roles,
        failure_observation,
        failed_h3_prompt=failed_prompt,
        repair_action=(
            "repair the exact Observer-detected failure and keep the corrected edited content "
            "locked to each mapped reference anchor while Video 1 remains authoritative for motion and timing"
        ),
        failure_type=failure_type,
    )
    prompt = str(composed.get("h3_prompt", "")).strip()
    if not prompt:
        raise ApimartError(f"task {task_id} Qwen returned an empty reference-aware prompt")
    prompt_path = bridge_dir / f"task_{task_id}_S1_prompt_only_optimized_h3_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    prompt_only = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(prompt_only, dict):
        raise ApimartError(f"task {task_id} prompt-only result is not an object")
    prompt_only.update({
        "reference_images_attached_to_final_prompt": references,
        "reference_aware_prompt_regenerated": True,
        "reference_aware_prompt_model": refiner.model,
        "reference_roles_attached_to_final_prompt": roles,
        "final_prompt_is_provisional_without_new_references": False,
        "optimized_h3_prompt_path": str(prompt_path),
        "final_refinement": composed,
    })
    write_json(result_path, prompt_only)
    return {
        "prompt_path": str(prompt_path),
        "reference_roles": roles,
        "final_refiner": composed,
    }


def generate_task(
    editor: GrsaiImageEditor,
    refiner: DashScopeVisionRefiner,
    plan_run: Path,
    out_dir: Path,
    task_id: int,
) -> dict[str, Any]:
    plan, result_path = load_plan(plan_run, task_id)
    result = plan["result"]
    source_video = source_video_from_plan(plan)
    raw_prompt = str(plan.get("raw_prompt", "")).strip()
    policy = str(result.get("reference_policy", "")).strip().lower()
    indices = result.get("anchor_frame_indices", [])
    image_prompts = result.get("image_edit_prompts", [])
    if policy not in {"video_only", "one_anchor", "three_anchor"}:
        raise ApimartError(f"task {task_id} has invalid reference policy: {policy}")
    if not isinstance(indices, list) or not isinstance(image_prompts, list):
        raise ApimartError(f"task {task_id} has invalid anchor data")
    expected = 0 if policy == "video_only" else 1 if policy == "one_anchor" else 3
    if len(indices) != expected or len(image_prompts) != expected:
        raise ApimartError(f"task {task_id} anchor/prompt count mismatch")

    task_root = out_dir / f"task_{task_id}" / "stages" / "S1"
    reference_dir = task_root / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    if policy == "video_only":
        record = {
            "kind": "fallback_reference_generation_v1",
            "task_id": str(task_id),
            "source_video": str(source_video),
            "reference_policy": policy,
            "reference_images": [],
            "image_editing_started": False,
            "minmax_h3_started": False,
            "status": "video_only_no_reference_required",
            "plan_result_path": str(result_path),
        }
        write_json(task_root / "reference_generation.json", record)
        return record

    references: list[str] = []
    states: list[dict[str, Any]] = []
    first_reference: Path | None = None
    for ordinal, (frame_value, prompt_value) in enumerate(zip(indices, image_prompts, strict=True), 1):
        if not isinstance(frame_value, int) or not isinstance(prompt_value, Mapping):
            raise ApimartError(f"task {task_id} has malformed image-edit entry {ordinal}")
        if prompt_value.get("frame_index") != frame_value:
            raise ApimartError(f"task {task_id} image-edit frame mapping is invalid")
        frame = context_frame(plan_run, task_id, source_video, frame_value)
        output = reference_dir / f"task_{task_id}_S1_reference_frame_{frame_value:03d}.png"
        state_path = reference_dir / f"task_{task_id}_S1_image_edit_state_frame_{frame_value:03d}.json"
        edit_prompt = str(prompt_value.get("prompt", "")).strip()
        if not edit_prompt:
            raise ApimartError(f"task {task_id} image-edit prompt {ordinal} is empty")
        style_reference = first_reference if policy == "three_anchor" and ordinal > 1 else None
        if output.is_file() and output.stat().st_size and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, Mapping) or state.get("status") != "succeeded":
                output.unlink(missing_ok=True)
                state_path.unlink(missing_ok=True)
            else:
                state = dict(state)
        if not output.is_file() or not output.stat().st_size:
            state = dict(editor.edit(
                frame,
                raw_prompt,
                edit_prompt,
                output,
                state_path,
                style_reference=style_reference,
            ))
        state.update({
            "frame_index": frame_value,
            "edit_input": str(frame),
            "content_output": str(output),
            "output": str(output),
            "style_reference": str(style_reference) if style_reference else None,
            "image_edit_prompt": edit_prompt,
        })
        write_json(state_path, state)
        references.append(str(output))
        states.append(state)
        if first_reference is None:
            first_reference = output
    prompt_result = compose_reference_aware_prompt(
        refiner,
        plan_run,
        task_id,
        plan,
        references,
        result_path,
    )
    record = {
        "kind": "fallback_reference_generation_v1",
        "task_id": str(task_id),
        "source_video": str(source_video),
        "reference_policy": policy,
        "anchor_frame_indices": indices,
        "reference_images": references,
        "image_edit_states": states,
        "image_editing_started": True,
        "minmax_h3_started": False,
        "status": "references_ready_before_minmax",
        "plan_result_path": str(result_path),
        "reference_aware_prompt_regenerated": True,
        "optimized_h3_prompt_path": prompt_result["prompt_path"],
        "reference_roles": prompt_result["reference_roles"],
    }
    write_json(task_root / "reference_generation.json", record)
    return record


def main() -> int:
    args = parse_args()
    tasks = selected_tasks(args)
    if not tasks:
        raise ApimartError("no tasks selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "selected_tasks.json", {
        "kind": "fallback_reference_generation_selection_v1",
        "plan_run": str(args.plan_run.resolve()),
        "tasks": tasks,
        "image_editing_started": False,
        "minmax_h3_started": False,
    })
    editor = GrsaiImageEditor(args.grsai_env)
    refiner = DashScopeVisionRefiner(
        Path("/root/.dashscope.env"),
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-vl-max",
        180,
    )
    summary_path = args.out_dir / "summary.json"
    summary: dict[str, Any] = {
        "kind": "fallback_reference_generation_v1",
        "plan_run": str(args.plan_run.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "tasks": [],
        "image_editing_started": False,
        "minmax_h3_started": False,
    }
    for task_id in tasks:
        try:
            record = generate_task(editor, refiner, args.plan_run, args.out_dir, task_id)
        except Exception as error:
            record = {"task_id": str(task_id), "status": "reference_generation_error", "error": str(error)}
        summary["tasks"] = [item for item in summary["tasks"] if item.get("task_id") != str(task_id)]
        summary["tasks"].append(record)
        if record.get("status") == "references_ready_before_minmax":
            summary["image_editing_started"] = True
        write_json(summary_path, summary)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary["status"] = "complete_before_minmax"
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
