#!/usr/bin/env python3
"""Run the Qwen-VL uniform-temporal Observer after fallback MiniMax outputs.

The fallback bundle has already produced each task's S1/output.mp4. This
script never submits H3 work: it waits for the existing output, compares it
with the normalized source video, and persists observation/observation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.media import is_aligned_video  # noqa: E402
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402
from apimart_h3_pipeline.providers.dashscope_client import DashScopeClient  # noqa: E402
from apimart_h3_pipeline.providers.vision_refiner import (  # noqa: E402
    DashScopeVisionRefiner,
    observe_stage_output,
)
from apimart_h3_pipeline.core.constants import POST_EDIT_OBSERVER_FRAME_INDICES  # noqa: E402


DEFAULT_TASKS = (16, 26, 30, 34, 35, 43, 44, 47, 49, 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_10_minmax_20260907_actual_refs"),
    )
    parser.add_argument("--tasks", nargs="+", type=int, default=list(DEFAULT_TASKS))
    parser.add_argument("--stage", default="S1")
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_inputs/compact_plans_content_only_qwen38_flash_final_20260906/compact_plans_without_camera_or_global_style"),
        help="Compact-plan directory used when a local fallback bundle has no bridge.json.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905"),
        help="Fallback root containing normalized task media when the bundle has no local copy.",
    )
    parser.add_argument("--dashscope-env", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument(
        "--dashscope-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--dashscope-model", default="qwen-vl-max")
    parser.add_argument("--dashscope-timeout", type=int, default=180)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    parser.add_argument("--force", action="store_true", help="rerun an existing Observer record")
    return parser.parse_args()


def raw_atomic_prompt(
    task_id: int,
    stage_label: str,
    bridge_path: Path,
    state_path: Path,
    plan_root: Path,
) -> tuple[str, str]:
    """Resolve the original stage instruction, never a generated H3 prompt."""

    if bridge_path.is_file():
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        value = bridge.get("next_raw_prompt")
        if isinstance(value, str) and value.strip():
            return value.strip(), "bridge_next_raw_prompt"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key in ("atomic_prompt", "raw_atomic_prompt", "next_raw_prompt"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"local_state_{key}"
        request = state.get("request")
        if isinstance(request, dict):
            for key in ("atomic_prompt", "raw_atomic_prompt"):
                value = request.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip(), f"local_state_request_{key}"
    plan_path = plan_root / f"task_{task_id:03d}.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rounds = plan.get("video_diffusion_rounds")
        if isinstance(rounds, list) and rounds:
            try:
                stage_number = max(1, int(str(stage_label).lstrip("S")))
            except ValueError:
                stage_number = 1
            item = rounds[min(stage_number - 1, len(rounds) - 1)]
            if isinstance(item, dict):
                value = item.get("instruction") or item.get("prompt")
                if isinstance(value, str) and value.strip():
                    return value.strip(), "compact_plan_video_diffusion_round"
    raise ApimartError(
        f"no raw atomic prompt found for task {task_id} {stage_label}; refusing to evaluate a generated H3 prompt"
    )


def wait_for_video(path: Path, timeout: int, interval: float) -> None:
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since = None
    while time.monotonic() < deadline:
        if path.is_file() and is_aligned_video(path):
            size = path.stat().st_size
            now = time.monotonic()
            if size == last_size:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= 15.0:
                    return
            else:
                last_size = size
                stable_since = now
        else:
            last_size = -1
            stable_since = None
        time.sleep(interval)
    raise ApimartError(f"timed out waiting for fallback output: {path}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    refiner = DashScopeVisionRefiner(
        args.dashscope_env,
        args.dashscope_base_url,
        args.dashscope_model,
        args.dashscope_timeout,
    )
    batch_path = run_dir / "observer_batch.json"
    batch = {
        "kind": "fallback_post_h3_observer_v2_uniform_temporal",
        "run_dir": str(run_dir),
        "stage": args.stage,
        "model": args.dashscope_model,
        "h3_submitted_by_this_script": False,
        "tasks": [],
    }
    for task_id in args.tasks:
        task_dir = run_dir / f"task_{task_id}"
        stage_dir = task_dir / "stages" / args.stage
        bridge_path = stage_dir / "bridge_for_next" / "bridge.json"
        state_path = stage_dir / "local_task_state.json"
        output = stage_dir / "output.mp4"
        source_video = task_dir / "media" / f"task_{task_id}_initial.mp4"
        if not source_video.is_file():
            source_video = args.source_root.resolve() / f"task_{task_id}" / "media" / f"task_{task_id}_initial.mp4"
        observation_path = stage_dir / "observation" / "observation.json"
        raw_prompt, prompt_provenance = raw_atomic_prompt(
            task_id, args.stage, bridge_path, state_path, args.plan_root.resolve()
        )
        if not source_video.is_file():
            raise ApimartError(f"missing normalized source video: {source_video}")
        reusable_observation = False
        if observation_path.is_file() and not args.force:
            record = json.loads(observation_path.read_text(encoding="utf-8"))
            reusable_observation = (
                record.get("kind") == "qwen_vl_uniform_temporal_success_gate_v2"
                and record.get("frame_indices") == list(POST_EDIT_OBSERVER_FRAME_INDICES)
                and record.get("prompt_provenance") not in {None, "unverified"}
            )
        if reusable_observation:
            status = "already_observed"
        else:
            print(json.dumps({"event": "fallback_observer_wait", "task": str(task_id), "output": str(output)}), flush=True)
            wait_for_video(output, args.wait_timeout, args.wait_seconds)
            print(json.dumps({"event": "fallback_observer_start", "task": str(task_id)}), flush=True)
            record = observe_stage_output(
                refiner,
                output,
                stage_dir,
                str(task_id),
                args.stage,
                raw_prompt,
                source_video,
                prompt_provenance,
            )
            status = "observed"
        item = {
            "task_id": str(task_id),
            "status": status,
            "success": record.get("success"),
            "failure_type": record.get("failure_type"),
            "confidence": record.get("confidence"),
            "observation": record.get("observation", ""),
            "prompt_provenance": record.get("prompt_provenance", prompt_provenance),
            "observation_path": str(observation_path),
            "output": str(output),
        }
        batch["tasks"].append(item)
        write_json(batch_path, batch)
        print(json.dumps({"event": "fallback_observer_complete", **item}, ensure_ascii=False), flush=True)
    batch["status"] = "complete"
    write_json(batch_path, batch)
    print(json.dumps({"event": "fallback_observer_batch_complete", "tasks": len(batch["tasks"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
