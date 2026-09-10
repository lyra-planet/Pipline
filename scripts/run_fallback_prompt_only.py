#!/usr/bin/env python3
"""Run semantic fallback Qwen planning/prompt generation without paid media.

The runner selects historical S1 Observer semantic failures, gives Qwen-VL the
raw requirement, failed H3 prompt, failure diagnosis, source checkpoints, and
any old references, then saves the new reference plan and final H3 prompt.
It deliberately does not call GRSAI image editing or MiniMax-H3.
"""
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
from apimart_h3_pipeline.core.policy import reference_policy  # noqa: E402
from apimart_h3_pipeline.core.repair_policy import (  # noqa: E402
    FailureDiagnosisAndRepair,
    RepairValidationError,
)
from apimart_h3_pipeline.media import select_keyframe  # noqa: E402
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.vision_refiner import DashScopeVisionRefiner  # noqa: E402


DEFAULT_SOURCE_RUN = Path("/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905")
DEFAULT_OUT_DIR = Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_50_prompt_only_20260908")
DEFAULT_EXCLUDE = [16, 26, 30, 34, 35, 43, 44, 47, 49, 56]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--exclude", nargs="*", type=int, default=DEFAULT_EXCLUDE)
    parser.add_argument("--task-list", type=Path)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When auto-selecting failures, skip tasks that already have a fallback_prompt_only.json in out-dir.",
    )
    parser.add_argument("--dashscope-env", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument(
        "--dashscope-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--dashscope-model", default="qwen-vl-max")
    parser.add_argument("--dashscope-timeout", type=int, default=180)
    return parser.parse_args()


def task_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"task_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_failure(source_task: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = source_task / "sequence_manifest.json"
    if not manifest_path.is_file():
        raise ApimartError(f"missing sequence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = next(
        (
            item
            for item in manifest.get("stages", [])
            if isinstance(item, Mapping) and item.get("stage") == "S1"
        ),
        None,
    )
    if not isinstance(stage, Mapping):
        raise ApimartError(f"task has no S1 stage: {manifest_path}")
    observation = stage.get("post_edit_observation") or stage.get("observation")
    if not isinstance(observation, Mapping) or observation.get("success") is not False:
        raise ApimartError("task is not a semantic S1 failure")
    failure_type = str(observation.get("failure_type", "")).strip().lower()
    if failure_type in {"observer_unavailable", "media_invalid", "not_frame_judgeable"}:
        raise ApimartError(f"task is not an actionable semantic failure: {failure_type}")
    raw_prompt = str(stage.get("raw_prompt", "")).strip()
    failed_prompt = str(stage.get("h3_prompt", "")).strip()
    diagnosis = stage.get("diagnosis")
    if isinstance(diagnosis, Mapping):
        failed_prompt = str(diagnosis.get("failed_prompt", failed_prompt)).strip()
    if not raw_prompt or not failed_prompt:
        raise ApimartError("failed task lacks raw_prompt or failed H3 prompt")
    source_video = source_task / "media" / f"{source_task.name}_initial.mp4"
    if not source_video.is_file():
        raise ApimartError(f"missing normalized initial video: {source_video}")
    return dict(stage), dict(observation), source_video


def choose_tasks(args: argparse.Namespace) -> list[int]:
    if args.count < 1:
        raise ApimartError("--count must be positive")
    if args.task_list:
        requested = [
            int(line.strip())
            for line in args.task_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return requested[: args.count]
    result: list[int] = []
    excluded = set(args.exclude)
    for task_dir in sorted(
        args.source_run.glob("task_*"),
        key=lambda item: task_id_from_path(item) if task_id_from_path(item) is not None else 10**9,
    ):
        task_id = task_id_from_path(task_dir)
        if task_id is None or task_id in excluded:
            continue
        if args.skip_existing and (
            args.out_dir / f"task_{task_id}" / "stages" / "S1" / "fallback_prompt_only.json"
        ).is_file():
            continue
        try:
            _, _, _ = load_failure(task_dir)
        except (ApimartError, OSError, ValueError, json.JSONDecodeError):
            continue
        result.append(task_id)
        if len(result) >= args.count:
            break
    return result


def make_repair(
    stage: Mapping[str, Any],
    observation: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    raw_prompt = str(stage.get("raw_prompt", "")).strip()
    failed_prompt = str(stage.get("h3_prompt", "")).strip()
    diagnosis_record = stage.get("diagnosis")
    if isinstance(diagnosis_record, Mapping):
        failed_prompt = str(diagnosis_record.get("failed_prompt", failed_prompt)).strip()
    policy = FailureDiagnosisAndRepair()
    try:
        diagnosis = policy.diagnose(
            observation,
            stage_id="S1",
            current_requirement=raw_prompt,
            failed_prompt=failed_prompt,
            attempt=1,
            evidence_frames=QWEN_CONTEXT_FRAME_INDICES,
        )
        repair = policy.repair(
            diagnosis,
            stage_id="S1",
            current_requirement=raw_prompt,
            failed_prompt=failed_prompt,
            retry_index=1,
            original_policy=reference_policy(raw_prompt, 1),
        )
    except RepairValidationError as error:
        # Prompt-only validation must not reject a semantic failure because a
        # local keyword guard cannot classify the raw request. Qwen receives
        # the original failure type/evidence and decides the actual repair.
        if "requires a motion cue" not in str(error):
            raise ApimartError(f"task {task_id} cannot build repair context: {error}") from error
        repair = {
            "kind": "qwen_prompt_only_repair_context_v1",
            "stage_id": "S1",
            "retry_index": 1,
            "failure_type": str(observation.get("failure_type", "unclassified")),
            "repair_action": "derive the corrective operation from the raw requirement and Observer evidence",
            "failed_prompt": failed_prompt,
            "observer_evidence": str(observation.get("observer_evidence", observation.get("observation", ""))),
            "reference_policy": "qwen_decide",
            "reference_image_count": -1,
        }
    return repair


def _reference_search_roots(source_task: Path, source_run: Path) -> list[Path]:
    """Return deterministic historical roots without assuming one run path."""

    roots = [
        source_task,
        source_task.parent,
        source_run.parent,
        Path("/root/autodl-tmp/Pipline_runs"),
        Path("/root/autodl-tmp"),
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            result.append(resolved)
    return result


def load_existing_references(
    source_task: Path,
    source_run: Path,
) -> tuple[list[Path], list[Mapping[str, Any]], dict[str, Any]]:
    """Load old references as Qwen evidence; never edit or overwrite them.

    Bridges historically stored absolute paths. If a run was copied or
    renamed, recover the same basename from the current task's attempts or a
    sibling historical bundle, but only when its task/stage path matches.
    """

    bridge_path = source_task / "stages" / "S1" / "bridge_for_next" / "bridge.json"
    if not bridge_path.is_file():
        return [], [], {"declared": [], "missing": [], "recovered": []}
    try:
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return [], [], {"declared": [], "missing": [], "recovered": []}
    declared = [str(item) for item in bridge.get("reference_images", []) if isinstance(item, str)]
    roles = bridge.get("reference_roles")
    declared_roles = roles if isinstance(roles, list) else []
    recovered: list[Path] = []
    missing: list[str] = []
    search_roots = _reference_search_roots(source_task, source_run)
    for item in declared:
        original = Path(item)
        candidates: list[Path] = []
        if original.is_file():
            candidates.append(original)
        basename = original.name
        if basename:
            for root in search_roots:
                try:
                    candidates.extend(sorted(root.rglob(basename)))
                except OSError:
                    continue
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
                and candidate.stat().st_size
                and re.search(rf"(?:^|/)task_{task_id_from_path(source_task) or 0}(?:/|$)", str(candidate))
                and "/stages/S1/" in str(candidate)
            ),
            None,
        )
        if selected is None:
            missing.append(item)
        elif selected not in recovered:
            recovered.append(selected)
    usable_roles = (
        [item for item in declared_roles if isinstance(item, Mapping)]
        if len(declared_roles) == len(recovered)
        and all(isinstance(item, Mapping) for item in declared_roles)
        else []
    )
    audit = {
        "declared": declared,
        "missing": missing,
        "recovered": [str(item) for item in recovered],
        "declared_roles": declared_roles,
    }
    return recovered, usable_roles, audit


def prepare_context(source_video: Path, bridge_dir: Path, task_id: int) -> list[Path]:
    context_frames = [
        bridge_dir / f"task_{task_id}_S1_context_frame_{frame_index:03d}.png"
        for frame_index in QWEN_CONTEXT_FRAME_INDICES
    ]
    for frame, frame_index in zip(context_frames, QWEN_CONTEXT_FRAME_INDICES, strict=True):
        if not frame.is_file() or not frame.stat().st_size:
            select_keyframe(source_video, frame, frame_index)
    return context_frames


def run_task(
    refiner: DashScopeVisionRefiner,
    source_run: Path,
    out_dir: Path,
    task_id: int,
) -> dict[str, Any]:
    source_task = source_run / f"task_{task_id}"
    stage, observation, source_video = load_failure(source_task)
    raw_prompt = str(stage["raw_prompt"]).strip()
    failed_prompt = str(stage.get("h3_prompt", "")).strip()
    if isinstance(stage.get("diagnosis"), Mapping):
        failed_prompt = str(stage["diagnosis"].get("failed_prompt", failed_prompt)).strip()
    repair = make_repair(stage, observation, task_id)
    target_stage = out_dir / f"task_{task_id}" / "stages" / "S1"
    bridge_dir = target_stage / "bridge_for_next"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    context_frames = prepare_context(source_video, bridge_dir, task_id)
    existing_refs, existing_roles, reference_audit = load_existing_references(source_task, source_run)
    failure_type = str(observation.get("failure_type", "unclassified"))
    failure_observation = str(
        observation.get("observer_evidence", observation.get("observation", ""))
    ).strip()

    repair_plan = refiner.plan_failure_repair(
        context_frames,
        existing_refs,
        raw_prompt,
        failed_prompt,
        failure_type,
        failure_observation,
        existing_roles,
    )
    plan_path = bridge_dir / f"task_{task_id}_S1_qwen_failure_repair_plan.json"
    write_json(plan_path, {
        "kind": "qwen_failure_repair_plan_prompt_only_v1",
        "source_task": str(source_task),
        "source_video": str(source_video),
        "raw_prompt": raw_prompt,
        "failed_h3_prompt": failed_prompt,
        "failure_type": failure_type,
        "failure_observation": failure_observation,
        "existing_reference_images": [str(item) for item in existing_refs],
        "existing_reference_roles": existing_roles,
        "existing_reference_audit": reference_audit,
        "result": repair_plan,
        "image_editing_started": False,
        "minmax_h3_started": False,
    })

    # No GRSAI output exists in this validation mode. Generate a provisional
    # prompt from source frames and diagnostics only; the final reference-aware
    # prompt will be regenerated after Qwen's chosen images are actually made.
    planned_count = int(repair_plan.get("reference_image_count", 0))
    final_references = existing_refs if len(existing_refs) == planned_count else []
    final_roles = existing_roles if len(existing_roles) == len(final_references) else []
    final_refinement = refiner.compose_h3_prompt(
        context_frames,
        final_references,
        raw_prompt,
        bool(stage.get("is_global_style", False)),
        final_roles,
        failure_observation,
        failed_h3_prompt=failed_prompt,
        repair_action=str(repair.get("repair_action", "targeted_repair")),
        failure_type=failure_type,
    )
    prompt_path = bridge_dir / f"task_{task_id}_S1_prompt_only_optimized_h3_prompt.txt"
    prompt_path.write_text(str(final_refinement["h3_prompt"]).strip() + "\n", encoding="utf-8")
    result_path = target_stage / "fallback_prompt_only.json"
    result = {
        "kind": "qwen_failure_fallback_prompt_only_v1",
        "task_id": str(task_id),
        "source_task": str(source_task),
        "source_video": str(source_video),
        "failure_type": failure_type,
        "reference_policy": repair_plan["reference_policy"],
        "planned_anchor_frame_indices": repair_plan["anchor_frame_indices"],
        "image_edit_prompts": repair_plan["image_edit_prompts"],
        "reference_images_used_for_planner": [str(item) for item in existing_refs],
        "reference_images_attached_to_final_prompt": [str(item) for item in final_references],
        "reference_audit": reference_audit,
        "reference_images_generated": [],
        "image_editing_started": False,
        "minmax_h3_started": False,
        "final_prompt_is_provisional_without_new_references": True,
        "repair_plan_path": str(plan_path),
        "optimized_h3_prompt_path": str(prompt_path),
    }
    write_json(result_path, result)
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = choose_tasks(args)
    if not tasks:
        raise ApimartError("no eligible semantic failures found")
    selection = {
        "kind": "fallback_prompt_only_task_selection_v1",
        "source_run": str(args.source_run.resolve()),
        "count": len(tasks),
        "excluded": args.exclude,
        "tasks": tasks,
        "image_editing_started": False,
        "minmax_h3_started": False,
    }
    selected_path = args.out_dir / "selected_tasks.json"
    if not selected_path.exists():
        write_json(selected_path, selection)
    refiner = DashScopeVisionRefiner(
        args.dashscope_env,
        args.dashscope_base_url,
        args.dashscope_model,
        args.dashscope_timeout,
    )
    summary_path = args.out_dir / "summary.json"
    existing_summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, dict) and loaded_summary.get("kind") == "qwen_failure_fallback_prompt_only_v1":
                existing_summary = loaded_summary
        except (OSError, ValueError, json.JSONDecodeError):
            existing_summary = {}
    summary: dict[str, Any] = existing_summary or {
        "kind": "qwen_failure_fallback_prompt_only_v1",
        "source_run": str(args.source_run.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "tasks": [],
        "image_editing_started": False,
        "minmax_h3_started": False,
    }
    summary["source_run"] = str(args.source_run.resolve())
    summary["out_dir"] = str(args.out_dir.resolve())
    summary["image_editing_started"] = False
    summary["minmax_h3_started"] = False
    existing_by_task = {
        str(item.get("task_id")): item
        for item in summary.get("tasks", [])
        if isinstance(item, Mapping) and item.get("task_id") is not None
    }
    task_order = [str(item.get("task_id")) for item in summary.get("tasks", []) if isinstance(item, Mapping)]
    for task_id in tasks:
        if str(task_id) not in task_order:
            task_order.append(str(task_id))
    for task_id in tasks:
        try:
            record = run_task(refiner, args.source_run, args.out_dir, task_id)
            record["status"] = "prompt_ready_before_image_editing"
        except Exception as error:  # keep one task failure from stopping the batch
            record = {"task_id": str(task_id), "status": "fallback_error", "error": str(error)}
        existing_by_task[str(task_id)] = record
        summary["tasks"] = [existing_by_task[item] for item in task_order if item in existing_by_task]
        write_json(summary_path, summary)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary["status"] = "complete_before_image_editing_and_minmax"
    write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
