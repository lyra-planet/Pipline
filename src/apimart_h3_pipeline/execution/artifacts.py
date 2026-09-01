"""Resumable H3 requests, attempt archives, and manifest helpers."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..providers.apimart import ApimartError, write_json
from ..core.repair_policy import RepairValidationError, validate_observation

from ..core.constants import H3_CLIENT_MODULE, H3_FRAME_COUNT, H3_FPS, H3_CANVAS_WIDTH, H3_CANVAS_HEIGHT
from ..media import CanvasGeometry, has_audio, is_aligned_video, is_h3_input_video, is_h3_generated_video, materialize_stage_video, read_json, source_canvas_geometry, write_geometry_sidecar
from ..core.policy import normalized_prompt
from ..providers.local import LocalH3Client, LocalH3Config

def reusable_h3_video_url(
    state_path: Path,
    h3_prompt: str,
    image_urls: Sequence[str],
) -> str | None:
    """Reuse a stage's exact source URL when resuming an existing H3 task."""

    if not state_path.is_file():
        return None
    try:
        state = read_json(state_path)
    except (OSError, json.JSONDecodeError, ApimartError):
        return None
    request = state.get("request")
    if not isinstance(request, Mapping):
        return None
    if request.get("prompt") != h3_prompt or request.get("images", []) != list(image_urls):
        return None
    references = request.get("reference_videos")
    if not isinstance(references, list) or len(references) != 1 or not isinstance(references[0], str):
        return None
    return references[0]


def invoke_h3_client(
    args: argparse.Namespace,
    stage: Mapping[str, str],
    h3_prompt: str,
    video_url: str,
    image_urls: Sequence[str],
    stage_dir: Path,
    bridge: Mapping[str, Any] | None = None,
) -> None:
    if getattr(args, "h3_backend", "online") == "local":
        reference_values = bridge.get("reference_images", []) if isinstance(bridge, Mapping) else []
        if not isinstance(reference_values, list) or not all(isinstance(item, str) for item in reference_values):
            raise ApimartError("local H3 bridge has invalid reference image paths")
        workflow_template = getattr(args, "local_workflow_template", None)
        if workflow_template is not None and not isinstance(workflow_template, Path):
            workflow_template = Path(str(workflow_template))
        if workflow_template is None:
            raise ApimartError("--local-workflow-template is required with --h3-backend local")
        local_client = LocalH3Client(LocalH3Config(
            server=getattr(args, "local_server", "http://127.0.0.1:8188"),
            workflow_template=workflow_template,
            input_dir=getattr(args, "local_input_dir", None) or args.out_dir / "local_inputs",
            output_dir=getattr(args, "local_output_dir", None) or args.out_dir / "local_outputs",
            timeout_seconds=getattr(args, "local_timeout", 21600),
            poll_seconds=getattr(args, "local_poll_seconds", 15.0),
        ))
        print(json.dumps({
            "event": "stage_start",
            "backend": "local",
            "stage": stage["stage_id"],
            "raw_prompt": stage["prompt"],
            "h3_prompt": h3_prompt,
            "reference_image_count": len(reference_values),
        }, ensure_ascii=False), flush=True)
        local_client.generate(
            source_video=Path(video_url),
            prompt=h3_prompt,
            reference_images=[Path(item) for item in reference_values],
            destination=stage_dir / "output.mp4",
            stage_dir=stage_dir,
            stage_id=str(stage["stage_id"]),
        )
        return
    command = [
        sys.executable, "-m", H3_CLIENT_MODULE, "--env-file", str(args.apimart_env),
        "--request-timeout", str(args.request_timeout), "--poll-seconds", str(args.poll_seconds),
        "--total-timeout", str(args.total_timeout), "generate", "--prompt", h3_prompt,
        "--model", args.h3_model,
        "--duration", str(args.duration), "--resolution", args.resolution, "--aspect-ratio", args.aspect_ratio,
        "--video-url", video_url, "--out-dir", str(stage_dir),
    ]
    for image_url in image_urls:
        command.extend(["--image-url", image_url])
    if args.allow_resubmit:
        command.append("--resubmit")
    print(json.dumps({
        "event": "stage_start",
        "stage": stage["stage_id"],
        "raw_prompt": stage["prompt"],
        "h3_prompt": h3_prompt,
        "reference_image_count": len(image_urls),
    }, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise ApimartError(f"APIMart H3 client failed for stage {stage['stage_id']} with exit code {completed.returncode}")


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, ApimartError):
        return {}


def attempt_number_from_path(path: Path) -> int | None:
    match = re.fullmatch(r"attempt_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_archived_attempts(stage_dir: Path) -> list[dict[str, Any]]:
    attempts_root = stage_dir / "attempts"
    if not attempts_root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for attempt_dir in sorted(
        (item for item in attempts_root.iterdir() if item.is_dir()),
        key=lambda item: attempt_number_from_path(item) or 0,
    ):
        attempt_number = attempt_number_from_path(attempt_dir)
        record = read_optional_json(attempt_dir / "attempt.json")
        if attempt_number is None or not record:
            continue
        record = dict(record)
        record.setdefault("attempt", attempt_number)
        if "post_edit_observation" not in record and isinstance(record.get("observation"), Mapping):
            record["post_edit_observation"] = record["observation"]
        records.append(record)
    return records


def load_current_observation(stage_dir: Path, stage_label: str) -> dict[str, Any] | None:
    candidate = read_optional_json(stage_dir / "observation" / "observation.json")
    if candidate.get("stage") != stage_label:
        return None
    try:
        return validate_observation(candidate)
    except RepairValidationError:
        return None


def confirmed_previous_requirements(
    manifest: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    current_index: int,
) -> list[dict[str, Any]]:
    """Return only completed parent-stage prompts for preservation repair."""

    saved = manifest.get("stages")
    if not isinstance(saved, list):
        return []
    prior_ids = {str(stage.get("stage_id")) for stage in stages[:current_index]}
    result: list[dict[str, Any]] = []
    for item in saved:
        if not isinstance(item, Mapping):
            continue
        stage_id = str(item.get("stage", item.get("stage_id", "")))
        if stage_id not in prior_ids:
            continue
        observation = item.get("post_edit_observation", item.get("observation"))
        success = observation.get("success") if isinstance(observation, Mapping) else None
        if success is True:
            prompt = normalized_prompt(str(item.get("raw_prompt", item.get("prompt", ""))))
            result.append({"stage_id": stage_id, "prompt": prompt, "status": "confirmed"})
    return result


def archive_stage_attempt(
    *,
    stage_dir: Path,
    attempt: int,
    output: Path,
    state_path: Path,
    bridge_dir: Path,
    observation_dir: Path,
    geometry: CanvasGeometry,
    media_dir: Path,
    task_id: str,
    stage_label: str,
    record: Mapping[str, Any],
) -> Path:
    """Move a failed attempt aside before issuing the next paid request."""

    archive = stage_dir / "attempts" / f"attempt_{attempt}"
    archive.mkdir(parents=True, exist_ok=True)
    first_output = archive / "output.mp4"
    if output.is_file():
        if first_output.exists():
            first_output.unlink()
        shutil.move(str(output), first_output)
    for source, target in (
        (state_path, archive / state_path.name),
        (stage_dir / "local_task_state.json", archive / "local_task_state.json"),
        (bridge_dir, archive / bridge_dir.name),
        (observation_dir, archive / observation_dir.name),
    ):
        if source.exists():
            if target.exists():
                if source.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(source), target)
    if first_output.is_file() and is_aligned_video(first_output):
        materialize_stage_video(
            first_output,
            media_dir / f"task_{task_id}_{stage_label}_attempt{attempt}.mp4",
            geometry,
        )
    archived_record = dict(record)
    if first_output.is_file():
        archived_record["output"] = str(first_output)
    archived_record["archive"] = str(archive)
    write_json(archive / "attempt.json", archived_record)
    return archive


def stage_failure_entry(
    *,
    stage_label: str,
    raw_prompt: str,
    attempts: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any],
    diagnosis: Mapping[str, Any] | None,
    repair: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": stage_label,
        "raw_prompt": raw_prompt,
        "status": "semantic_failure" if observation.get("success") is False else "observation_pending",
        "output": None,
        "post_edit_observation": dict(observation),
        "diagnosis": dict(diagnosis) if diagnosis else None,
        "repair": dict(repair) if repair else None,
        "attempts": [dict(item) for item in attempts],
    }


def replace_manifest_stage(manifest: dict[str, Any], entry: Mapping[str, Any]) -> None:
    stage_id = entry.get("stage")
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        stages = []
    manifest["stages"] = [
        item for item in stages
        if not isinstance(item, Mapping) or item.get("stage") != stage_id
    ]
    manifest["stages"].append(dict(entry))
