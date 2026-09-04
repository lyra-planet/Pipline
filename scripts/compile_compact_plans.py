#!/usr/bin/env python3
"""Compile ZIP compact plans with their source-video index into runner jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--ordered-plans", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_by_task: dict[str, str] = {}
    with args.ordered_plans.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id", ""))
            source = str(record.get("source_video", ""))
            if task_id and source:
                source_by_task[task_id] = source

    jobs: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(args.compact_dir.glob("task_*.json")):
        compact = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(compact.get("task_id", ""))
        source_value = source_by_task.get(task_id)
        if not source_value:
            errors.append(f"{path.name}: source video is absent from ordered_plans")
            continue
        source_name = Path(source_value).name
        source = (args.video_root / source_name).resolve()
        if not source.is_file() or source.stat().st_size == 0:
            errors.append(f"{path.name}: source video missing: {source}")
            continue
        rounds = compact.get("video_diffusion_rounds")
        if not isinstance(rounds, list) or not rounds:
            errors.append(f"{path.name}: video_diffusion_rounds is empty")
            continue
        stages: list[dict[str, str]] = []
        for index, item in enumerate(rounds, 1):
            prompt = item.get("instruction") if isinstance(item, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{path.name}: round {index} has no instruction")
                break
            stages.append({
                "stage_id": f"S{index}",
                "audited_content_only_prompt": prompt.strip(),
            })
        if len(stages) != len(rounds):
            continue
        jobs.append({
            "task_id": task_id,
            "source_video": str(source),
            "sequential_nominal_plan": stages,
        })

    if errors:
        raise SystemExit("\n".join(errors))
    ids = [str(job["task_id"]) for job in jobs]
    if len(jobs) != 626 or len(set(ids)) != len(ids):
        raise SystemExit(f"expected 626 unique tasks, found {len(jobs)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"tasks": jobs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(jobs), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
