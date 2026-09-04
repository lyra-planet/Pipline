"""Stage-oriented APIMart H3 execution entry point.

The runner owns only lifecycle orchestration; provider and media details live in
neighboring modules so the package can be moved and tested independently.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from ..providers.apimart import ApimartClient, ApimartError, resolve_credentials, write_json
from ..core.repair_policy import FailureDiagnosisAndRepair, RepairValidationError, STAGE_RETRY_LIMIT, fixed_three_anchor_repair, global_style_three_anchor_repair, stage_outcome, validate_repair_record

from .artifacts import archive_stage_attempt, attempt_number_from_path, confirmed_previous_requirements, invoke_h3_client, load_archived_attempts, load_current_observation, read_json, read_optional_json, replace_manifest_stage, reusable_h3_video_url, stage_failure_entry
from ..bridge import bridge_for_stage, deterministic_repair_h3_prompt, load_task, public_url, three_anchor_reference_plan
from ..core.constants import DEFAULT_STATIC_REFERENCE_COUNT, H3_CANVAS_HEIGHT, H3_CANVAS_WIDTH, PROMPT_KEY, QWEN_CONTEXT_FRAME_INDICES
from ..media import CanvasGeometry, geometry_sidecar, has_audio, is_aligned_video, is_h3_input_video, load_geometry_sidecar, materialize_final_video, materialize_initial_video, materialize_stage_video, source_canvas_geometry, write_geometry_sidecar
from ..core.policy import reference_policy
from ..providers.vision_refiner import DashScopeVisionRefiner, observe_stage_output
from ..providers.grsai import GrsaiImageEditor
from ..providers.local import LocalH3MediaAdapter

from .cli import parse_args


def next_parent_video_url(
    apimart: Any,
    parent: Path,
    saved_next_url: Any,
    media_public_base_url: str | None,
    backend: str,
) -> str:
    """Resolve the parent reference persisted after a completed stage.

    Online APIMart stages consume a public media URL, while the local ComfyUI
    backend consumes filesystem paths directly. Keeping this decision in one
    helper prevents local runs from accidentally calling ``public_url`` with a
    missing base URL or with a caller-supplied URL that local H3 cannot read.
    """

    if backend == "local":
        return str(parent.resolve())
    if (
        apimart.is_ctmoai
        and isinstance(saved_next_url, str)
        and saved_next_url.startswith(apimart.base_url.rstrip("/") + "/sd-media/")
    ):
        return saved_next_url
    if apimart.is_ctmoai:
        return apimart.upload_media(parent, "videos")
    if not media_public_base_url:
        raise ApimartError("--media-public-base-url is required for online parent videos")
    return public_url(media_public_base_url, parent.name)


def main() -> int:
    args = parse_args()
    backend = getattr(args, "h3_backend", "online")
    if backend not in {"online", "local"}:
        raise ApimartError(f"unsupported H3 backend: {backend}")
    if backend == "local":
        workflow_template = getattr(args, "local_workflow_template", None)
        if workflow_template is not None and not isinstance(workflow_template, Path):
            workflow_template = Path(str(workflow_template))
            args.local_workflow_template = workflow_template
        if workflow_template is None or not workflow_template.is_file():
            raise ApimartError("--local-workflow-template must name an existing file with --h3-backend local")
    if not 4 <= args.duration <= 15:
        raise ApimartError("duration must be from 4 to 15")
    task = load_task(args.compiled_jobs.resolve(), args.task_id)
    start_stage = getattr(args, "start_stage", None)
    start_index = 0
    if start_stage:
        start_index = next((index for index, stage in enumerate(task["stages"]) if stage["stage_id"] == start_stage), None)
        if start_index is None:
            valid = ", ".join(stage["stage_id"] for stage in task["stages"])
            raise ApimartError(f"--start-stage must be one of: {valid}")
    if args.last_stage:
        stage_index = next((index for index, stage in enumerate(task["stages"]) if stage["stage_id"] == args.last_stage), None)
        if stage_index is None:
            valid = ", ".join(stage["stage_id"] for stage in task["stages"])
            raise ApimartError(f"--last-stage must be one of: {valid}")
        task["stages"] = task["stages"][:stage_index + 1]
    if start_index >= len(task["stages"]):
        raise ApimartError("--start-stage must not be after --last-stage")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "sequence_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {
        "kind": "apimart_minimax_h3_online_qwen_refined_sequential_v8",
        "task_id": task["task_id"],
        "source_video": str(task["source_video"]),
        "duration": args.duration,
        "resolution": args.resolution,
        "h3_model": args.h3_model,
        "aspect_ratio": args.aspect_ratio,
        "stages": [],
    }
    manifest["kind"] = "apimart_minimax_h3_online_qwen_refined_sequential_v8"
    manifest["prompt_refiner"] = {
        "provider": "dashscope",
        "model": args.dashscope_model,
        "input_contract": (
            "one raw atomic prompt plus five parent-video frames; Qwen observes all five while frame 0 is the "
            "deterministic primary style master, sees every generated reference, and every completed stage is "
            "checked on five output frames"
        ),
        "global_style_contract": (
            "global visual-style stages use a frame-0 primary style master on the normal attempt and "
            "escalate to frame-0/frame-53/frame-106 temporal anchors only after a semantic failure; "
            "ordinary static stages use "
            f"{args.global_style_reference_count} reference image(s); video owns motion only"
        ),
        "structured_plan_visibility": False,
    }
    prepared_initial = args.prepared_initial_video.resolve() if args.prepared_initial_video else None
    if prepared_initial is not None:
        if not prepared_initial.is_file() or not is_aligned_video(prepared_initial) or not is_h3_input_video(prepared_initial):
            raise ApimartError(f"prepared initial video is not 1344x768/107-frame/24fps: {prepared_initial}")
        if not has_audio(prepared_initial):
            raise ApimartError(f"prepared initial video has no audio track: {prepared_initial}")
        try:
            geometry = load_geometry_sidecar(prepared_initial)
        except ApimartError:
            # A normalized output from an earlier stage is also a valid H3
            # canvas when launching a targeted rerun. Its sidecar carries the
            # stage_input role rather than initial_input.
            geometry = load_geometry_sidecar(prepared_initial, role="stage_input")
    else:
        geometry = source_canvas_geometry(task["source_video"])
    manifest["normalization"] = {
        "contract": "letterbox_to_h3_canvas_then_crop_final_output",
        "h3_canvas": {"width": H3_CANVAS_WIDTH, "height": H3_CANVAS_HEIGHT},
        "source_geometry": geometry.as_dict(),
        "intermediate_inputs": "all stage inputs are 1344x768, 107 frames, 24 fps",
        "final_output": "crop to the fitted content rectangle; no geometric stretch",
    }
    initial_target = args.media_dir / f"task_{task['task_id']}_initial.mp4"
    if prepared_initial is not None:
        if not initial_target.is_file() or initial_target.stat().st_size != prepared_initial.stat().st_size:
            shutil.copy2(prepared_initial, initial_target)
        write_geometry_sidecar(initial_target, geometry, "initial_input")
        initial_media = initial_target
    else:
        initial_media = materialize_initial_video(task["source_video"], initial_target, geometry)
    parent = initial_media
    if backend == "online" and not args.media_public_base_url:
        raise ApimartError("--media-public-base-url is required with --h3-backend online")
    parent_url = (
        public_url(args.media_public_base_url, parent.name)
        if backend == "online"
        else str(parent.resolve())
    )
    if start_index:
        previous_stage = task["stages"][start_index - 1]["stage_id"] if start_index < len(task["stages"]) else f"S{start_index}"
        previous_output = args.out_dir / "stages" / previous_stage / "output.mp4"
        if not previous_output.is_file() or not is_aligned_video(previous_output):
            raise ApimartError(
                f"cannot start at {args.start_stage}: completed output for {previous_stage} is missing or invalid: {previous_output}"
            )
        parent = previous_output
        parent_url = (
            str(parent.resolve())
            if backend == "local"
            else public_url(args.media_public_base_url, parent.name)
        )
        task["stages"] = task["stages"][start_index:]
    if args.dry_run:
        for stage in task["stages"]:
            manifest["stages"].append({"stage": stage["stage_id"], "prompt": stage["prompt"], "policy": reference_policy(stage["prompt"], args.global_style_reference_count)})
        write_json(manifest_path, manifest)
        print(json.dumps({"event": "dry_run", "manifest": str(manifest_path)}, ensure_ascii=False))
        return 0
    if backend == "online":
        credentials_args = argparse.Namespace(env_file=args.apimart_env, base_url=None)
        api_key, base_url = resolve_credentials(credentials_args)
        apimart = ApimartClient(api_key, base_url, args.request_timeout)
    else:
        apimart = LocalH3MediaAdapter()
    if apimart.is_ctmoai:
        saved_initial_url = manifest.get("initial_video_url")
        if isinstance(saved_initial_url, str) and saved_initial_url.startswith(
            apimart.base_url.rstrip("/") + "/sd-media/"
        ):
            parent_url = saved_initial_url
        else:
            parent_url = apimart.upload_media(parent, "videos")
        manifest["initial_video_url"] = parent_url
        write_json(manifest_path, manifest)
    refiner = DashScopeVisionRefiner(
        args.dashscope_env,
        args.dashscope_base_url,
        args.dashscope_model,
        args.dashscope_timeout,
    )
    editor = GrsaiImageEditor(args.grsai_env)
    repair_policy = FailureDiagnosisAndRepair()
    manifest["failure_recovery"] = {
        "mode": args.failure_recovery,
        "stage_retry_limit": STAGE_RETRY_LIMIT,
        "allow_unverified_output": bool(args.allow_unverified_output),
        "diagnosis_schema": "qwen_vl_failure_diagnosis_v1",
        "repair_schema": "vetra_failure_repair_v1",
    }
    manifest["h3_backend"] = backend
    if backend == "local":
        manifest["local_h3"] = {
            "server": args.local_server,
            "workflow_template": str(args.local_workflow_template.resolve()) if args.local_workflow_template else None,
            "input_dir": str((args.local_input_dir or args.out_dir / "local_inputs").resolve()),
            "output_dir": str((args.local_output_dir or args.out_dir / "local_outputs").resolve()),
        }
    write_json(manifest_path, manifest)
    for index, stage in enumerate(task["stages"]):
        stage_dir = args.out_dir / "stages" / stage["stage_id"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_label = stage["stage_id"]
        output = stage_dir / "output.mp4"
        bridge_dir = stage_dir / "bridge_for_next"
        state_path = stage_dir / "apimart_task_state.json"
        raw_prompt = stage["prompt"]
        base_policy = reference_policy(raw_prompt, args.global_style_reference_count)
        # A stage's parent is immutable across semantic retries. A failed H3
        # output is a candidate for observation/archive only; it must never
        # become the source video for another request in the same stage.
        stage_parent_video = parent
        stage_parent_url = parent_url
        attempts = load_archived_attempts(stage_dir)
        retry_index = len(attempts)
        if retry_index > STAGE_RETRY_LIMIT:
            raise ApimartError(
                f"stage {stage_label} has {retry_index} archived retries; "
                f"the fixed retry limit is {STAGE_RETRY_LIMIT}"
            )
        repair_context: dict[str, Any] | None = None
        observation_pending = False
        propagated_failed_output = False
        observer_skipped_on_retry = False
        if attempts:
            prior = attempts[-1]
            saved_repair = prior.get("repair")
            if isinstance(saved_repair, Mapping):
                try:
                    repair_context = validate_repair_record(
                        saved_repair,
                        stage_id=stage_label,
                        current_requirement=raw_prompt,
                    )
                except RepairValidationError as error:
                    raise ApimartError(f"archived repair is invalid for {stage_label}: {error}") from error
            elif int(prior.get("reference_image_count", 0) or 0) == 1:
                # Resume a pre-VETRA run that archived a single-reference
                # failure before entering its next retry.  A saved global
                # style policy takes the targeted temporal-anchor route;
                # archives without policy metadata retain legacy behavior.
                prior_observation = prior.get("post_edit_observation", prior.get("observation", {}))
                evidence = str(prior_observation.get("observation", "")) if isinstance(prior_observation, Mapping) else ""
                try:
                    prior_policy = prior.get("policy")
                    if isinstance(prior_policy, Mapping) and prior_policy.get("is_global_style") is True:
                        repair_context = global_style_three_anchor_repair(
                            stage_id=stage_label,
                            current_requirement=raw_prompt,
                            failed_prompt=str(prior.get("h3_prompt", raw_prompt)),
                            observer_evidence=evidence,
                            failure_type=str(
                                prior_observation.get("failure_type", "unclassified")
                            ) if isinstance(prior_observation, Mapping) else "unclassified",
                            retry_index=retry_index,
                        )
                    else:
                        repair_context = fixed_three_anchor_repair(
                            stage_id=stage_label,
                            current_requirement=raw_prompt,
                            failed_prompt=str(prior.get("h3_prompt", raw_prompt)),
                            observer_evidence=evidence,
                            retry_index=retry_index,
                        )
                except RepairValidationError as error:
                    raise ApimartError(f"legacy fallback cannot be resumed for {stage_label}: {error}") from error

        while True:
            bridge_failure_observation = (
                str(repair_context.get("observer_evidence", ""))
                if repair_context is not None else None
            )
            image_urls, h3_prompt, bridge = bridge_for_stage(
                refiner,
                editor,
                apimart,
                stage_parent_video,
                stage_dir,
                raw_prompt,
                # This argument controls only the ordinary static policy.
                # A repair context carries its own count, including zero for
                # video-only motion repairs, and bridge_for_stage applies it
                # after validating the ordinary policy.
                args.global_style_reference_count,
                args.media_public_base_url,
                args.media_dir,
                task["task_id"],
                bridge_failure_observation,
                repair_context,
                geometry=geometry,
            )
            if apimart.is_ctmoai:
                reusable_url = reusable_h3_video_url(state_path, h3_prompt, image_urls)
                if reusable_url is not None and reusable_url != stage_parent_url:
                    raise ApimartError(
                        f"persisted H3 request for {stage_label} points to a different parent video; "
                        "failed stage output cannot be used as retry input"
                    )
            stage_input_url = stage_parent_url
            invoke_h3_client(
                args,
                stage,
                h3_prompt,
                stage_input_url,
                image_urls,
                stage_dir,
                bridge=bridge,
            )
            if not is_aligned_video(output):
                raise ApimartError(f"completed stage output is invalid: {output}")

            # A retry is the bounded repair attempt for this stage.  Its output
            # must use the immutable stage parent above, but it is deliberately
            # not sent through Observer again: the retry result is propagated
            # as the next stage's parent and remains explicitly marked as
            # unverified in the manifest.
            if retry_index > 0:
                retry_observation = {
                    "success": None,
                    "failure_type": "observer_skipped_retry",
                    "observation": "Retry output propagated without a second Observer call.",
                    "confidence": 0.0,
                }
                attempt_record: dict[str, Any] = {
                    "attempt": retry_index + 1,
                    "reference_image_count": len(image_urls),
                    "h3_input_video": str(stage_parent_video),
                    "h3_input_video_url": stage_input_url,
                    "h3_prompt": h3_prompt,
                    "output": str(output),
                    "observation": retry_observation,
                    "post_edit_observation": retry_observation,
                    "observer_skipped": True,
                    # Preserve the diagnosis that authorized this retry so
                    # the stage-level manifest remains actionable even though
                    # this final attempt has no new Observer diagnosis.
                    "diagnosis": attempts[-1].get("diagnosis") if attempts else None,
                    "repair": repair_context,
                    "policy": base_policy,
                }
                attempts.append(attempt_record)
                observation = retry_observation
                propagated_failed_output = True
                observer_skipped_on_retry = True
                break

            observation = load_current_observation(stage_dir, stage_label)
            if observation is None:
                observation = observe_stage_output(
                    refiner,
                    output,
                    stage_dir,
                    task["task_id"],
                    stage_label,
                    raw_prompt,
                )
            attempt_record: dict[str, Any] = {
                "attempt": retry_index + 1,
                "reference_image_count": len(image_urls),
                "h3_input_video": str(stage_parent_video),
                "h3_input_video_url": stage_input_url,
                "h3_prompt": h3_prompt,
                "output": str(output),
                "observation": observation,
                "post_edit_observation": observation,
                "repair": repair_context,
                "policy": base_policy,
            }
            attempts.append(attempt_record)

            outcome = stage_outcome(observation, allow_unverified=args.allow_unverified_output)
            if outcome == "success":
                break
            if outcome in {"observation_pending", "unverified_success"}:
                observation_pending = True
                entry = stage_failure_entry(
                    stage_label=stage_label,
                    raw_prompt=raw_prompt,
                    attempts=attempts,
                    observation=observation,
                    diagnosis=None,
                    repair=repair_context,
                )
                entry["output"] = str(output)
                entry["h3_prompt"] = h3_prompt
                replace_manifest_stage(manifest, entry)
                manifest["status"] = "observation_pending"
                write_json(manifest_path, manifest)
                if outcome == "observation_pending":
                    raise ApimartError(
                        f"Observer unavailable for {stage_label}; refusing to propagate unverified output"
                    )
                break

            try:
                diagnosis = repair_policy.diagnose(
                    observation,
                    stage_id=stage_label,
                    current_requirement=raw_prompt,
                    failed_prompt=h3_prompt,
                    previous_requirements=confirmed_previous_requirements(manifest, task["stages"], index),
                    attempt=retry_index + 1,
                    evidence_frames=QWEN_CONTEXT_FRAME_INDICES,
                )
            except RepairValidationError as error:
                raise ApimartError(f"failure diagnosis rejected for {stage_label}: {error}") from error
            attempt_record["diagnosis"] = diagnosis
            global_style_fallback = (
                bool(base_policy.get("is_global_style"))
                and len(image_urls) == 1
            )
            retry_allowed = (
                args.failure_recovery != "disabled"
                and retry_index < STAGE_RETRY_LIMIT
                and (
                    diagnosis.get("repairable") is True
                    or global_style_fallback
                    or (
                        args.failure_recovery == "fixed-three-anchor"
                        and bool(base_policy.get("needs_reference_image"))
                        and len(image_urls) == 1
                    )
                )
            )
            if not retry_allowed:
                entry = stage_failure_entry(
                    stage_label=stage_label,
                    raw_prompt=raw_prompt,
                    attempts=attempts,
                    observation=observation,
                    diagnosis=diagnosis,
                    repair=repair_context,
                )
                entry["output"] = str(output)
                entry["h3_prompt"] = h3_prompt
                replace_manifest_stage(manifest, entry)
                manifest["status"] = "semantic_failure"
                write_json(manifest_path, manifest)
                raise ApimartError(
                    f"semantic failure at {stage_label}: "
                    f"{diagnosis.get('failure_type', 'unclassified')} after {retry_index} retries"
                )

            try:
                if args.failure_recovery == "fixed-three-anchor":
                    next_repair = fixed_three_anchor_repair(
                        stage_id=stage_label,
                        current_requirement=raw_prompt,
                        failed_prompt=h3_prompt,
                        observer_evidence=str(diagnosis.get("observer_evidence", "")),
                        retry_index=retry_index + 1,
                    )
                elif global_style_fallback:
                    next_repair = global_style_three_anchor_repair(
                        stage_id=stage_label,
                        current_requirement=raw_prompt,
                        failed_prompt=h3_prompt,
                        observer_evidence=str(diagnosis.get("observer_evidence", "")),
                        failure_type=str(diagnosis.get("failure_type", "unclassified")),
                        retry_index=retry_index + 1,
                    )
                else:
                    next_repair = repair_policy.repair(
                        diagnosis,
                        stage_id=stage_label,
                        current_requirement=raw_prompt,
                        failed_prompt=h3_prompt,
                        previous_requirements=confirmed_previous_requirements(manifest, task["stages"], index),
                        retry_index=retry_index + 1,
                        original_policy=base_policy,
                    )
            except RepairValidationError as error:
                raise ApimartError(f"repair policy rejected {stage_label}: {error}") from error
            attempt_record["repair"] = next_repair
            archive = archive_stage_attempt(
                stage_dir=stage_dir,
                attempt=retry_index + 1,
                output=output,
                state_path=state_path,
                bridge_dir=bridge_dir,
                observation_dir=stage_dir / "observation",
                geometry=geometry,
                media_dir=args.media_dir,
                task_id=task["task_id"],
                stage_label=stage_label,
                record=attempt_record,
            )
            attempt_record["archive"] = str(archive)
            print(json.dumps({
                "event": "stage_targeted_repair" if args.failure_recovery == "targeted" else "reference_escalation",
                "stage": stage_label,
                "attempt": retry_index + 1,
                "failure_type": diagnosis.get("failure_type"),
                "repair_action": next_repair.get("repair_action"),
                "reference_policy": next_repair.get("reference_policy"),
                "reason": diagnosis.get("observer_evidence", "edit_not_confirmed"),
            }, ensure_ascii=False), flush=True)
            retry_index += 1
            repair_context = next_repair

        if not is_aligned_video(output):
            raise ApimartError(f"completed stage output is invalid after observation: {output}")
        media_output = materialize_stage_video(
            output,
            args.media_dir / f"task_{task['task_id']}_{stage_label}.mp4",
            geometry,
        )
        parent = media_output
        saved_next_url = next(
            (
                item.get("next_parent_video_url")
                for item in manifest.get("stages", [])
                if isinstance(item, Mapping)
                and item.get("stage") == stage_label
                and item.get("output") == str(output)
            ),
            None,
        )
        parent_url = next_parent_video_url(
            apimart,
            parent,
            saved_next_url,
            args.media_public_base_url,
            backend,
        )
        reference_escalated = any(int(item.get("reference_image_count", 0) or 0) == 3 for item in attempts)
        entry = {
            "stage": stage_label,
            "raw_prompt": raw_prompt,
            "status": (
                "semantic_failure_propagated"
                if propagated_failed_output
                else "observation_pending" if observation_pending else "success"
            ),
            "h3_prompt": h3_prompt,
            "output": str(output),
            "media": str(media_output),
            "h3_input_video_url": stage_input_url,
            "next_parent_video_url": parent_url,
            "has_reference_image": bool(image_urls),
            "reference_image_count": len(image_urls),
            "reused_existing_output": False,
            "post_edit_observation": observation,
            "reference_escalated": reference_escalated,
            "attempts": attempts,
            "diagnosis": attempts[-1].get("diagnosis"),
            "repair": repair_context,
            "propagated_failed_output": propagated_failed_output,
            "observer_skipped": observer_skipped_on_retry,
        }
        if bridge:
            entry["bridge"] = bridge
        replace_manifest_stage(manifest, entry)
        manifest["status"] = (
            "degraded" if propagated_failed_output
            else "observation_pending" if observation_pending else "running"
        )
        write_json(manifest_path, manifest)
        print(json.dumps({"event": "stage_complete", "stage": stage_label, "output": str(output), "reference_escalated": reference_escalated, "attempts": len(attempts)}, ensure_ascii=False), flush=True)
    final = args.out_dir / "output.mp4"
    final_geometry = geometry_sidecar(final)
    final_metadata = read_json(final_geometry) if final_geometry.is_file() else {}
    if (
        not final.is_file()
        or not is_aligned_video(final)
        or final_metadata.get("role") != "final_output"
        or final_metadata.get("geometry") != geometry.as_dict()
    ):
        materialize_final_video(parent, final, geometry)
    manifest["output"] = str(final)
    has_propagated_failure = any(
        isinstance(item, Mapping) and item.get("propagated_failed_output") is True
        for item in manifest.get("stages", [])
    )
    manifest["status"] = (
        "degraded" if has_propagated_failure
        else "observation_pending"
        if any(
            isinstance(item, Mapping) and item.get("status") == "observation_pending"
            for item in manifest.get("stages", [])
        ) else "success"
    )
    write_json(manifest_path, manifest)
    print(json.dumps({"event": "sequence_complete", "output": str(final)}, ensure_ascii=False))
    return 0
