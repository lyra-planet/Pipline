#!/usr/bin/env python3
"""Run semantic fallback through Qwen/GRSAI and stop before MiniMax-H3.

This is a bridge-only validation runner. It consumes historical S1 Observer
failures, asks Qwen to choose the fallback topology and write the repaired H3
prompt, creates any references with GRSAI, and persists the bridge. It never
submits a MiniMax-H3 request.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.bridge.execution import bridge_for_stage  # noqa: E402
from apimart_h3_pipeline.core.constants import QWEN_CONTEXT_FRAME_INDICES  # noqa: E402
from apimart_h3_pipeline.core.policy import reference_policy  # noqa: E402
from apimart_h3_pipeline.core.repair_policy import (  # noqa: E402
    FailureDiagnosisAndRepair,
    RepairValidationError,
)
from apimart_h3_pipeline.media import (  # noqa: E402
    load_geometry_sidecar,
    source_canvas_geometry,
    write_geometry_sidecar,
)
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.grsai import GrsaiImageEditor  # noqa: E402
from apimart_h3_pipeline.providers.vision_refiner import DashScopeVisionRefiner  # noqa: E402


class BridgeOnlyMedia:
    """Provider marker used with a local public-media copy, never H3."""

    is_ctmoai = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_50_bridge_only_20260907"),
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--exclude",
        nargs="*",
        type=int,
        default=[16, 26, 30, 34, 35, 43, 44, 47, 49, 56],
    )
    parser.add_argument("--dashscope-env", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument("--grsai-env", type=Path, default=Path("/root/.grsai.env"))
    parser.add_argument(
        "--dashscope-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--dashscope-model", default="qwen-vl-max")
    parser.add_argument("--dashscope-timeout", type=int, default=180)
    parser.add_argument("--media-base-url", default="http://127.0.0.1:19999/media")
    parser.add_argument("--task-list", type=Path)
    return parser.parse_args()


def task_id_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"task_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_failure(source_task: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest_path = source_task / "sequence_manifest.json"
    if not manifest_path.is_file():
        raise ApimartError(f"missing sequence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = next((item for item in manifest.get("stages", []) if isinstance(item, Mapping) and item.get("stage") == "S1"), None)
    if not isinstance(stage, Mapping):
        raise ApimartError(f"task has no S1 stage: {manifest_path}")
    observation = stage.get("post_edit_observation") or stage.get("observation")
    if not isinstance(observation, Mapping) or observation.get("success") is not False:
        raise ApimartError("task is not a semantic S1 failure")
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
    return dict(stage), dict(observation), source_video, manifest


def choose_tasks(args: argparse.Namespace) -> list[int]:
    if args.task_list:
        tasks = [int(line.strip()) for line in args.task_list.read_text().splitlines() if line.strip()]
        return tasks[: args.count]
    result: list[int] = []
    excluded = set(args.exclude)
    for task_dir in sorted(args.source_run.glob("task_*"), key=lambda p: task_id_from_path(p) or 10**9):
        task_id = task_id_from_path(task_dir)
        if task_id is None or task_id in excluded:
            continue
        try:
            _, observation, _, _ = load_failure(task_dir)
        except (ApimartError, OSError, ValueError, json.JSONDecodeError):
            continue
        failure_type = str(observation.get("failure_type", ""))
        if failure_type in {"observer_unavailable", "media_invalid"}:
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
    diagnosis = stage.get("diagnosis")
    if isinstance(diagnosis, Mapping):
        failed_prompt = str(diagnosis.get("failed_prompt", failed_prompt)).strip()
    if not failed_prompt:
        raise ApimartError(f"task {task_id} has no failed H3 prompt")
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
        raise ApimartError(f"task {task_id} cannot build repair context: {error}") from error
    return repair


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = choose_tasks(args)
    if not tasks:
        raise ApimartError("no eligible semantic failures found")
    write_json(args.out_dir / "selected_tasks.json", {
        "kind": "fallback_bridge_only_task_selection_v1",
        "source_run": str(args.source_run.resolve()),
        "count": len(tasks),
        "excluded": args.exclude,
        "tasks": tasks,
        "minmax_h3_started": False,
        "observer_after_h3": "pending_until_minmax_output",
    })
    refiner = DashScopeVisionRefiner(
        args.dashscope_env,
        args.dashscope_base_url,
        args.dashscope_model,
        args.dashscope_timeout,
    )
    editor = GrsaiImageEditor(args.grsai_env)
    media = BridgeOnlyMedia()
    summary: dict[str, Any] = {
        "kind": "qwen_grsai_fallback_bridge_only_v2",
        "source_run": str(args.source_run.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "tasks": [],
        "minmax_h3_started": False,
        "observer_after_h3": "not_run_without_h3_output",
    }
    summary_path = args.out_dir / "summary.json"
    for task_id in tasks:
        source_task = args.source_run / f"task_{task_id}"
        target_task = args.out_dir / f"task_{task_id}"
        target_stage = target_task / "stages" / "S1"
        try:
            stage, observation, source_video, manifest = load_failure(source_task)
            raw_prompt = str(stage["raw_prompt"]).strip()
            repair = make_repair(stage, observation, task_id)
            target_media = target_task / "media"
            target_media.mkdir(parents=True, exist_ok=True)
            target_video = target_media / source_video.name
            if not target_video.exists() or target_video.stat().st_size != source_video.stat().st_size:
                shutil.copy2(source_video, target_video)
            try:
                geometry = load_geometry_sidecar(source_video)
            except ApimartError:
                geometry = source_canvas_geometry(source_video)
                write_geometry_sidecar(target_video, geometry, "initial_input")
            bridge_for_stage(
                refiner,
                editor,
                media,
                target_video,
                target_stage,
                raw_prompt,
                global_style_reference_count=1,
                media_public_base_url=args.media_base_url,
                media_dir=target_media,
                task_id=str(task_id),
                failure_observation=str(observation.get("observer_evidence", observation.get("observation", ""))),
                repair_context=repair,
                geometry=geometry,
            )
            bridge = json.loads((target_stage / "bridge_for_next" / "bridge.json").read_text(encoding="utf-8"))
            record = {
                "task_id": str(task_id),
                "status": "fallback_ready_before_minmax",
                "failure_type": observation.get("failure_type"),
                "reference_policy": (bridge.get("qwen_failure_repair_plan") or {}).get("reference_policy"),
                "reference_count": len(bridge.get("reference_images", [])),
                "h3_prompt_path": bridge.get("optimized_h3_prompt_path"),
                "bridge_path": str(target_stage / "bridge_for_next" / "bridge.json"),
                "observer_after_h3": "pending_minmax_output",
            }
        except Exception as error:  # continue the batch and preserve the failure
            record = {"task_id": str(task_id), "status": "fallback_error", "error": str(error)}
        summary["tasks"].append(record)
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
