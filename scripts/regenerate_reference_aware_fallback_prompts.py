#!/usr/bin/env python3
"""Regenerate fallback H3 prompts after actual reference images exist."""
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
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.vision_refiner import DashScopeVisionRefiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-run", type=Path, required=True)
    parser.add_argument("--references-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dashscope-env", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument("--dashscope-model", default="qwen-vl-max")
    parser.add_argument("--dashscope-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--task-list", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApimartError(f"expected JSON object: {path}")
    return value


def roles_for(plan: Mapping[str, Any], reference_count: int) -> list[dict[str, Any]]:
    indices = plan.get("anchor_frame_indices")
    if not isinstance(indices, list) or len(indices) != reference_count:
        raise ApimartError("reference count does not match Qwen anchor frame indices")
    if reference_count == 1:
        return [{"picture_index": 1, "role": "edited primary anchor", "source_frame_index": int(indices[0])}]
    if reference_count == 3:
        return [
            {"picture_index": 1, "role": "edited start anchor", "source_frame_index": int(indices[0])},
            {"picture_index": 2, "role": "edited middle anchor", "source_frame_index": int(indices[1])},
            {"picture_index": 3, "role": "edited end anchor", "source_frame_index": int(indices[2])},
        ]
    raise ApimartError(f"unsupported reference count: {reference_count}")


def task_ids(path: Path) -> list[int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        value = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if isinstance(value, Mapping):
        value = value.get("tasks", [])
    if not isinstance(value, list):
        raise ApimartError(f"task list is not a list: {path}")
    return [int(item) for item in value if str(item).isdigit()]


def regenerate(args: argparse.Namespace, refiner: DashScopeVisionRefiner, task_id: int) -> dict[str, Any]:
    task_root = args.plan_run / f"task_{task_id}" / "stages" / "S1"
    bridge_dir = task_root / "bridge_for_next"
    result_path = task_root / "fallback_prompt_only.json"
    plan_path = bridge_dir / f"task_{task_id}_S1_qwen_failure_repair_plan.json"
    if not result_path.is_file() or not plan_path.is_file():
        raise ApimartError(f"missing prompt-only artifacts for task {task_id}")
    result = load_object(result_path)
    plan = load_object(plan_path)
    qwen_result = plan.get("result")
    if not isinstance(qwen_result, Mapping):
        raise ApimartError(f"missing Qwen repair result for task {task_id}")
    reference_record_path = args.references_run / f"task_{task_id}" / "stages" / "S1" / "reference_generation.json"
    reference_record = load_object(reference_record_path) if reference_record_path.is_file() else {}
    references = [Path(str(item)) for item in reference_record.get("reference_images", [])]
    if references and not all(item.is_file() and item.stat().st_size for item in references):
        raise ApimartError(f"task {task_id} has missing reference image files")
    context_frames = [
        bridge_dir / f"task_{task_id}_S1_context_frame_{frame_index:03d}.png"
        for frame_index in QWEN_CONTEXT_FRAME_INDICES
    ]
    if not all(item.is_file() and item.stat().st_size for item in context_frames):
        raise ApimartError(f"task {task_id} has missing source context frames")
    raw_prompt = str(plan.get("raw_prompt", "")).strip()
    failed_prompt = str(plan.get("failed_h3_prompt", "")).strip()
    failure_type = str(plan.get("failure_type", "unclassified")).strip()
    failure_observation = str(plan.get("failure_observation", "")).strip()
    roles = roles_for(qwen_result, len(references)) if references else []
    refined = refiner.compose_h3_prompt(
        context_frames,
        references,
        raw_prompt,
        False,
        roles,
        failure_observation,
        failed_h3_prompt=failed_prompt,
        repair_action="repair the exact Observer-detected failure and keep the corrected edited content locked to the mapped reference anchors across the sequence",
        failure_type=failure_type,
    )
    prompt = str(refined.get("h3_prompt", "")).strip()
    if not prompt:
        raise ApimartError(f"Qwen returned an empty prompt for task {task_id}")
    labels = sorted(set(map(int, re.findall(r"<Picture\s*(\d+)>", prompt, re.I))))
    expected_labels = list(range(1, len(references) + 1))
    if references and labels != expected_labels:
        raise ApimartError(f"Qwen prompt for task {task_id} omitted picture labels: {labels} != {expected_labels}")
    prompt_path = bridge_dir / f"task_{task_id}_S1_prompt_only_optimized_h3_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    result["reference_images_attached_to_final_prompt"] = [str(item) for item in references]
    result["final_prompt_is_provisional_without_new_references"] = False
    result["reference_aware_prompt_regenerated"] = True
    result["reference_aware_prompt_model"] = refiner.model
    result["reference_roles_attached_to_final_prompt"] = roles
    result["optimized_h3_prompt_path"] = str(prompt_path)
    result["final_refinement"] = refined
    write_json(result_path, result)
    return {"task_id": str(task_id), "status": "regenerated", "reference_count": len(references), "prompt": str(prompt_path)}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    refiner = DashScopeVisionRefiner(args.dashscope_env, args.dashscope_base_url, args.dashscope_model, 180)
    records: list[dict[str, Any]] = []
    for task_id in task_ids(args.task_list):
        try:
            record = regenerate(args, refiner, task_id)
        except Exception as error:
            record = {"task_id": str(task_id), "status": "error", "error": str(error)}
        records.append(record)
        write_json(args.out_dir / "summary.json", {"kind": "reference_aware_prompt_regeneration_v1", "tasks": records})
        print(json.dumps(record, ensure_ascii=False), flush=True)
    failed = [item for item in records if item.get("status") == "error"]
    write_json(args.out_dir / "summary.json", {
        "kind": "reference_aware_prompt_regeneration_v1",
        "status": "complete" if not failed else "complete_with_errors",
        "tasks": records,
    })
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
