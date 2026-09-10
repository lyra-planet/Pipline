#!/usr/bin/env python3
"""Use Qwen to classify and split camera/style rounds from compact plans."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = r"""Classify one video-editing plan round. Return JSON only with exactly:
{"camera_motion": boolean, "global_style": boolean, "reason": string}

camera_motion=true only when this round requests CHANGING the camera/viewpoint,
shot framing, or camera movement. This includes pan, zoom, dolly, tilt, orbit,
tracking, changing camera movement/speed, changing a static shot, re-framing,
shot scale, or camera angle/perspective. If the instruction only says to keep or
preserve an existing camera pan, shot, viewpoint, or static framing unchanged,
camera_motion=false. If only a subject/object is moved within the frame and no
camera change is requested, camera_motion=false. If an actual camera change is
combined with another edit in the same round, camera_motion=true.
Examples: "Move the bird to the center of the frame" is subject repositioning
and camera_motion=false; "Move the camera to center the bird" is camera_motion=true.
"Keep the existing pan unchanged" is camera_motion=false.

global_style=true only when the round explicitly requests a scene-wide visual
style, art medium, aesthetic, film treatment, or color-grade treatment for the
whole scene/video/frame. Examples include applying watercolor, oil painting,
anime, cartoon, a cyberpunk visual style/aesthetic, vintage film, sepia,
monochrome documentary, sketch, claymation, LEGO, or a clearly scene-wide
cinematic style. Do NOT count a background/scene/environment replacement or a
new location merely described as futuristic/cyberpunk (for example, changing
the scene into a cyberpunk city), weather, season, ordinary lighting changes,
environmental VFX/particles, or a generic atmosphere change as global_style
unless the instruction explicitly frames it as a scene-wide visual style or art
treatment. A named whole-video rendering or motion treatment such as
stop-motion counts as global style when it changes the video's overall visual
or temporal treatment, even if the phrase does not literally say "entire
scene". Object-only color/material/clothing
changes, a person's facial expression or appearance, a lens look, a wide-shot
look, a local icon, a background phrase such as "look out", and subject/object
actions are global_style=false. If a round contains a qualifying global style
edit together with another edit, global_style=true.
Examples that are NOT global style: replacing a background or location with a
cyberpunk city, changing season or weather, changing ordinary scene lighting,
adding environmental VFX, applying depth of field, or moving an object within
the frame. These remain content/composition edits unless an explicit scene-wide
art/style treatment is requested.

Judge the actual semantics, not whether a particular verb such as transform,
change, or apply appears. Do not infer an edit that is not requested."""


def load_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("DASHSCOPE_API_KEY", "")


def parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Qwen response is not an object")
    if not isinstance(value.get("camera_motion"), bool):
        raise ValueError("camera_motion is not boolean")
    if not isinstance(value.get("global_style"), bool):
        raise ValueError("global_style is not boolean")
    if not isinstance(value.get("reason"), str):
        raise ValueError("reason is not a string")
    return value


def classify(key: str, model: str, task_id: str, original: str, round_number: int,
             instruction: str, timeout: int, retries: int) -> dict[str, Any]:
    user = json.dumps({
        "task_id": task_id,
        "original_complex_instruction": original,
        "round": round_number,
        "instruction": instruction,
    }, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=body, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            return parse_json(result["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Qwen classification failed: {last}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    key = load_key(args.env_file)
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is missing")
    paths = sorted(source.glob("task_*.json"))
    if not paths:
        raise SystemExit(f"no task JSON files found: {source}")
    items: list[tuple[Path, dict[str, Any], int, dict[str, Any]]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rounds = data.get("video_diffusion_rounds")
        if not isinstance(rounds, list):
            raise SystemExit(f"{path.name}: rounds is not a list")
        for index, item in enumerate(rounds, 1):
            if not isinstance(item, dict) or not isinstance(item.get("instruction"), str):
                raise SystemExit(f"{path.name}: invalid round {index}")
            items.append((path, data, index, item))
    def one(entry: tuple[Path, dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
        path, data, index, item = entry
        result = classify(key, args.model, str(data.get("task_id", "")),
                          str(data.get("original_complex_instruction", "")), index,
                          item["instruction"], args.timeout, args.retries)
        return {"task_id": str(data.get("task_id", "")), "source_file": path.name,
                "original_round": int(item.get("round", index)),
                "instruction": item["instruction"], **result}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one, entry) for entry in items]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
            if index % 100 == 0 or index == len(futures):
                print(f"classified {index}/{len(futures)}", flush=True)
    if errors:
        raise SystemExit(f"classification errors: {len(errors)}; first: {errors[0]}")
    by_file: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_file.setdefault(record["source_file"], []).append(record)
    plan_out = output / "compact_plans_without_camera_or_global_style"
    plan_out.mkdir(parents=True, exist_ok=True)
    camera: list[dict[str, Any]] = []
    style: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        records_for_file = sorted(by_file[path.name], key=lambda x: x["original_round"])
        kept: list[dict[str, Any]] = []
        for record, item in zip(records_for_file, data["video_diffusion_rounds"], strict=True):
            if record["camera_motion"]:
                camera.append(record)
            if record["global_style"]:
                style.append(record)
            if record["camera_motion"] or record["global_style"]:
                combined.append({**record, "categories": (["camera_motion"] if record["camera_motion"] else []) + (["global_style"] if record["global_style"] else [])})
            else:
                kept.append(item)
        for new_index, item in enumerate(kept, 1):
            item["round"] = new_index
        output_data = dict(data)
        output_data["video_diffusion_rounds"] = kept
        (plan_out / path.name).write_text(json.dumps(output_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stats.append({"task_id": str(data.get("task_id", "")), "source_file": path.name,
                      "original_round_count": len(data["video_diffusion_rounds"]),
                      "remaining_round_count": len(kept),
                      "removed_round_count": len(data["video_diffusion_rounds"]) - len(kept)})
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = source.parent / "compact_plans_qwen38_flash_applied_repairs.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, output / source_manifest.name)
    for name, values in (("camera_motion_rounds.json", camera), ("global_style_rounds.json", style), ("camera_or_global_style_rounds.json", combined)):
        (output / name).write_text(json.dumps({"model": args.model, "records": values, "record_count": len(values)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"source": str(source), "output": str(output), "model": args.model,
                "classifier": "Qwen JSON classification per round", "no_reorder": True,
                "rounds_are_renumbered_after_removal": True,
                "original_round_numbers_preserved_in_extracted_records": True,
                "task_count": len(paths), "round_count": len(items),
                "camera_motion_round_count": len(camera), "global_style_round_count": len(style),
                "combined_removed_round_count": len(combined),
                "tasks_with_removed_rounds": sum(1 for x in stats if x["removed_round_count"]),
                "tasks_with_no_remaining_rounds": sum(1 for x in stats if x["remaining_round_count"] == 0),
                "task_stats": stats}
    (output / "extraction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("task_count", "round_count", "camera_motion_round_count", "global_style_round_count", "combined_removed_round_count", "tasks_with_no_remaining_rounds", "output")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
