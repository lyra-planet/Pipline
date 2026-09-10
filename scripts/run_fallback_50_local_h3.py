#!/usr/bin/env python3
"""Run the saved 50-task fallback bundle through local MiniMax-H3.

The Qwen repair plans and GRSAI reference outputs are inputs to this runner;
this script does not call either service and never changes the saved topology.
It waits for reference generation to finish, then submits each task to one
local ComfyUI H3 server with resumable per-task state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.media import is_aligned_video  # noqa: E402
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.local import LocalH3Client, LocalH3Config  # noqa: E402
from apimart_h3_pipeline.bridge.helpers import validate_final_h3_prompt_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-run", type=Path, required=True)
    parser.add_argument("--references-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--workflow-template", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--comfy-output-dir", type=Path, required=True)
    parser.add_argument("--task-list", type=Path, help="Optional file containing one numeric task ID per line")
    parser.add_argument("--all-prompt-tasks", action="store_true", help="Discover every prompt-only task")
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--task-attempts", type=int, default=3)
    parser.add_argument("--atomic-no-ref", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApimartError(f"expected JSON object: {path}")
    return value


def wait_for_reference_task(args: argparse.Namespace, task_id: int) -> dict[str, Any]:
    final_path = args.plan_run / f"task_{task_id}" / "stages" / "S1" / "fallback_prompt_only.json"
    if final_path.is_file():
        final = read_json(final_path)
        if final.get("final_refinement") and not final.get("final_prompt_is_provisional_without_new_references", True):
            return {"reference_images": final.get("reference_images_attached_to_final_prompt", []), "reference_roles": final.get("reference_roles_attached_to_final_prompt", []), "source_video": final.get("source_video", "")}
    # Prefer the durable per-task record. The batch summary is intentionally
    # incremental and may not contain tasks that were completed before this
    # process started.
    task_record_path = args.references_run / f"task_{task_id}" / "stages" / "S1" / "reference_generation.json"
    if task_record_path.is_file():
        try:
            record = read_json(task_record_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ApimartError):
            record = {}
        if record.get("status") in {"references_ready_before_minmax", "video_only_no_reference_required"}:
            return record
    summary_path = args.references_run / "summary.json"
    while True:
        if summary_path.is_file():
            try:
                summary = read_json(summary_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ApimartError):
                summary = {}
            records = {
                int(str(item["task_id"])): item
                for item in summary.get("tasks", [])
                if isinstance(item, Mapping) and str(item.get("task_id", "")).isdigit()
            }
            if task_id in records:
                record = records[task_id]
                if record.get("status") == "reference_generation_error":
                    raise ApimartError(f"reference generation failed for task {task_id}: {record.get('error')}")
                return record
        print(json.dumps({"event": "waiting_for_task_reference", "task": str(task_id), "recorded": len(records) if 'records' in locals() else 0}, ensure_ascii=False), flush=True)
        time.sleep(max(5, args.wait_seconds))


def task_ids_from_selection(plan_run: Path, task_list: Path | None = None, all_prompt_tasks: bool = False) -> list[int]:
    if task_list is not None:
        values = []
        text = task_list.read_text(encoding="utf-8").replace("\\n", "\n")
        for line in text.splitlines():
            value = line.strip()
            if value and value.isdigit():
                values.append(int(value))
        if not values:
            raise ApimartError(f"task list has no numeric IDs: {task_list}")
        return sorted(set(values))
    if all_prompt_tasks:
        discovered: list[int] = []
        for result_path in plan_run.glob("task_*/stages/S1/fallback_prompt_only.json"):
            task_dir = result_path.parents[2]
            name = task_dir.name
            if name.startswith("task_") and name[5:].isdigit():
                discovered.append(int(name[5:]))
        if not discovered:
            raise ApimartError(f"no prompt-only task results found in {plan_run}")
        return sorted(set(discovered))
    selection = read_json(plan_run / "selected_tasks.json")
    tasks = selection.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ApimartError("prompt-only selection has no tasks")
    result = [int(item) for item in tasks if str(item).isdigit()]
    if len(result) != len(tasks):
        raise ApimartError("prompt-only selection contains invalid task IDs")
    return sorted(set(result))


def load_task_inputs(
    args: argparse.Namespace,
    task_id: int,
    reference_record: Mapping[str, Any],
) -> tuple[Path, str, list[Path]]:
    prompt_result_path = args.plan_run / f"task_{task_id}" / "stages" / "S1" / "fallback_prompt_only.json"
    if not prompt_result_path.is_file():
        raise ApimartError(f"missing prompt-only result: {prompt_result_path}")
    prompt_result = read_json(prompt_result_path)
    task_root = args.plan_run / f"task_{task_id}" / "stages" / "S1" / "bridge_for_next"
    prompt_path = task_root / f"task_{task_id}_S1_prompt_only_optimized_h3_prompt.txt"
    if not prompt_path.is_file():
        raise ApimartError(f"missing final fallback H3 prompt: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if args.atomic_no_ref:
        atomic = str(prompt_result.get("atomic_prompt", "")).strip()
        if not atomic:
            atomic = prompt
        prompt = "summary: [video editing] The target video is an edited version of <Video 1>. Reference images, if present, are optional visual references only. " + atomic
    if not prompt:
        raise ApimartError(f"empty final fallback H3 prompt: {prompt_path}")
    source_video = Path(str(reference_record.get("source_video", "")))
    if not source_video.is_file():
        source_video = Path("/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905") / f"task_{task_id}" / "media" / f"task_{task_id}_initial.mp4"
    if not source_video.is_file():
        raise ApimartError(f"missing source video for task {task_id}: {source_video}")
    refs = [] if args.atomic_no_ref else [Path(str(item)) for item in reference_record.get("reference_images", [])]
    if not all(item.is_file() and item.stat().st_size for item in refs):
        raise ApimartError("a reference used by the final Qwen prompt is missing; regenerate that prompt before submitting")
    if refs and prompt_result.get("reference_aware_prompt_regenerated") is not True:
        raise ApimartError(
            f"task {task_id} has no final reference-aware prompt; refusing to run H3 with a provisional prompt"
        )
    reference_roles = reference_record.get("reference_roles")
    if not isinstance(reference_roles, list):
        reference_roles = prompt_result.get("reference_roles_attached_to_final_prompt", [])
    final_refiner = prompt_result.get("final_refinement")
    validate_final_h3_prompt_contract(
        h3_prompt=prompt,
        final_refiner=final_refiner,
        reference_images=refs,
        reference_roles=reference_roles,
        expected_reference_count=len(refs),
        require_reference_aware_regeneration=bool(refs),
    )
    return source_video, prompt, refs


def run_task(
    client: LocalH3Client,
    args: argparse.Namespace,
    task_id: int,
    source_video: Path,
    prompt: str,
    references: list[Path],
) -> dict[str, Any]:
    stage_dir = args.out_dir / f"task_{task_id}" / "stages" / "S1"
    output = stage_dir / "output.mp4"
    record: dict[str, Any] = {
        "task_id": str(task_id),
        "source_video": str(source_video),
        "prompt_path": str(args.plan_run / f"task_{task_id}" / "stages" / "S1" / "bridge_for_next" / f"task_{task_id}_S1_prompt_only_optimized_h3_prompt.txt"),
        "reference_images": [str(item) for item in references],
        "reference_count": len(references),
        "minmax_h3_started": False,
        "output": str(output),
    }
    if output.is_file() and is_aligned_video(output):
        record["status"] = "already_complete"
        record["minmax_h3_started"] = True
        return record
    last_error: str | None = None
    for attempt in range(1, max(1, args.task_attempts) + 1):
        try:
            record["attempt"] = attempt
            record["minmax_h3_started"] = True
            print(json.dumps({"event": "local_h3_start", "task": str(task_id), "attempt": attempt, "reference_count": len(references)}, ensure_ascii=False), flush=True)
            client.generate(
                source_video=source_video,
                prompt=prompt,
                reference_images=references,
                destination=output,
                stage_dir=stage_dir,
                stage_id="S1",
            )
            if not output.is_file() or not is_aligned_video(output):
                raise ApimartError(f"invalid local H3 output: {output}")
            record["status"] = "completed"
            return record
        except Exception as error:
            last_error = str(error)
            record["last_error"] = last_error
            record["status"] = "retrying" if attempt < max(1, args.task_attempts) else "error"
            write_json(stage_dir / "local_batch_error.json", record)
            print(json.dumps({"event": "local_h3_error", "task": str(task_id), "attempt": attempt, "error": last_error}, ensure_ascii=False), flush=True)
    return record


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = task_ids_from_selection(args.plan_run, args.task_list, args.all_prompt_tasks)
    client = LocalH3Client(LocalH3Config(
        server=args.server,
        workflow_template=args.workflow_template,
        input_dir=args.input_dir,
        output_dir=args.comfy_output_dir,
        timeout_seconds=21600,
        poll_seconds=15.0,
    ))
    batch_path = args.out_dir / ("minmax_batch_" + "_".join(map(str, task_ids)) + ".json")
    batch: dict[str, Any] = {
        "kind": "fallback_qwen_50_local_minmax_h3_v1",
        "plan_run": str(args.plan_run.resolve()),
        "references_run": str(args.references_run.resolve()),
        "server": args.server,
        "workflow_template": str(args.workflow_template.resolve()),
        "qwen_called": False,
        "grsai_called": False,
        "minmax_h3_started": True,
        "tasks": [],
    }
    for task_id in task_ids:
        existing_output = args.out_dir / f"task_{task_id}" / "stages" / "S1" / "output.mp4"
        if existing_output.is_file() and is_aligned_video(existing_output):
            result = {
                "task_id": str(task_id),
                "status": "already_complete",
                "output": str(existing_output),
                "minmax_h3_started": True,
            }
            batch["tasks"] = [item for item in batch["tasks"] if item.get("task_id") != str(task_id)]
            batch["tasks"].append(result)
            write_json(batch_path, batch)
            print(json.dumps({"event": "local_h3_skip_existing", "task": str(task_id), "output": str(existing_output)}, ensure_ascii=False), flush=True)
            continue
        try:
            reference_record = wait_for_reference_task(args, task_id)
            source_video, prompt, refs = load_task_inputs(args, task_id, reference_record)
            result = run_task(client, args, task_id, source_video, prompt, refs)
        except Exception as error:
            result = {"task_id": str(task_id), "status": "input_error", "error": str(error), "minmax_h3_started": False}
            print(json.dumps({"event": "local_h3_input_error", "task": str(task_id), "error": str(error)}, ensure_ascii=False), flush=True)
        batch["tasks"] = [item for item in batch["tasks"] if item.get("task_id") != str(task_id)]
        batch["tasks"].append(result)
        write_json(batch_path, batch)
    batch["status"] = "complete"
    write_json(batch_path, batch)
    print(json.dumps({"event": "local_h3_batch_complete", "tasks": len(batch["tasks"])}, ensure_ascii=False), flush=True)
    return 1 if any(item.get("status") in {"input_error", "error"} for item in batch["tasks"]) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
