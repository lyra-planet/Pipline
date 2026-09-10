#!/usr/bin/env python3
"""Run the already-built fallback bridges through local MiniMax-H3 only.

This intentionally does not call Qwen-VL or GRSAI.  It consumes the bridge
and actual reference images prepared in a separate fallback run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apimart_h3_pipeline.media import is_aligned_video  # noqa: E402
from apimart_h3_pipeline.providers.local import LocalH3Client, LocalH3Config  # noqa: E402
from apimart_h3_pipeline.providers.apimart import ApimartError, write_json  # noqa: E402


TASKS = (16, 26, 30, 34, 35, 43, 44, 47, 49, 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/root/autodl-tmp/Pipline_runs/fallback_qwen_10_minmax_20260907_actual_refs"),
    )
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--workflow-template", type=Path, default=Path("/root/autodl-tmp/Pipline_inputs/temp/minimax_h3_ref2va_api.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("/root/autodl-tmp/ComfyUI/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/ComfyUI/output_gpu0"))
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = LocalH3Client(LocalH3Config(
        server=args.server,
        workflow_template=args.workflow_template,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
        poll_seconds=args.poll_seconds,
    ))
    run_dir = args.run_dir.resolve()
    batch = {
        "kind": "fallback_qwen_10_local_minmax_h3_v1",
        "run_dir": str(run_dir),
        "server": args.server,
        "workflow_template": str(args.workflow_template.resolve()),
        "qwen_called": False,
        "grsai_called": False,
        "minmax_h3_started": True,
        "tasks": [],
    }
    batch_path = run_dir / "minmax_batch.json"
    for task_id in TASKS:
        task_dir = run_dir / f"task_{task_id}"
        stage_dir = task_dir / "stages" / "S1"
        bridge_path = stage_dir / "bridge_for_next" / "bridge.json"
        if not bridge_path.is_file():
            raise ApimartError(f"missing fallback bridge: {bridge_path}")
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        prompt = bridge.get("h3_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApimartError(f"empty fallback H3 prompt for task {task_id}")
        source_video = task_dir / "media" / f"task_{task_id}_initial.mp4"
        if not source_video.is_file():
            raise ApimartError(f"missing normalized source video: {source_video}")
        references = [Path(item) for item in bridge.get("reference_images", [])]
        if not all(item.is_file() and item.stat().st_size for item in references):
            raise ApimartError(f"missing fallback reference image for task {task_id}")
        destination = stage_dir / "output.mp4"
        if destination.is_file() and is_aligned_video(destination):
            status = "already_complete"
        else:
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(json.dumps({
                "event": "minmax_fallback_start",
                "task": str(task_id),
                "reference_count": len(references),
                "prompt_path": bridge.get("optimized_h3_prompt_path"),
            }, ensure_ascii=False), flush=True)
            client.generate(
                source_video=source_video,
                prompt=prompt,
                reference_images=references,
                destination=destination,
                stage_dir=stage_dir,
                stage_id="S1",
            )
            if not destination.is_file() or not is_aligned_video(destination):
                raise ApimartError(f"MiniMax-H3 output is invalid for task {task_id}: {destination}")
            status = "completed"
        batch["tasks"].append({
            "task_id": str(task_id),
            "status": status,
            "source_video": str(source_video),
            "reference_images": [str(item) for item in references],
            "prompt_path": bridge.get("optimized_h3_prompt_path"),
            "output": str(destination),
        })
        write_json(batch_path, batch)
        print(json.dumps({"event": "minmax_fallback_complete", "task": str(task_id), "status": status, "output": str(destination)}, ensure_ascii=False), flush=True)
    batch["status"] = "complete"
    write_json(batch_path, batch)
    print(json.dumps({"event": "minmax_fallback_batch_complete", "tasks": len(batch["tasks"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
