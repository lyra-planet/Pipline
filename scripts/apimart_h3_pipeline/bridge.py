"""Reference-image bridge: parent frames -> image edits -> H3 prompt."""
from __future__ import annotations

import json
import shutil
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from run_apimart_minimax_h3 import ApimartClient, ApimartError, write_json
except ModuleNotFoundError:
    from ..run_apimart_minimax_h3 import ApimartClient, ApimartError, write_json

try:
    from vetra_failure_repair import RepairValidationError, apply_repair_clause, validate_repair_record
except ModuleNotFoundError:
    from ..vetra_failure_repair import RepairValidationError, apply_repair_clause, validate_repair_record

from .constants import (
    BRIDGE_KIND, DEFAULT_STATIC_REFERENCE_COUNT, GLOBAL_STYLE_REFERENCE_COUNT, PRIMARY_REFERENCE_FRAME_INDEX,
    QWEN_CONTEXT_FRAME_INDICES, TEMPORAL_END_FRAME_INDEX, TEMPORAL_MIDDLE_FRAME_INDEX, PROMPT_KEY,
)
from .image_editor import GrsaiImageEditor
from .media import select_keyframe
from .media import read_json
from .artifacts import attempt_number_from_path
from .policy import expected_reference_roles, normalized_prompt, reference_policy, reference_roles_match_policy, validate_h3_reference_tags
from .vision import DashScopeVisionRefiner

def load_task(compiled_jobs: Path, task_id: str) -> dict[str, Any]:
    document = read_json(compiled_jobs)
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ApimartError("compiled jobs document lacks tasks list")
    for task in tasks:
        if isinstance(task, Mapping) and str(task.get("task_id")) == task_id:
            stages = task.get("sequential_nominal_plan")
            source = Path(str(task.get("source_video", ""))).resolve()
            if not isinstance(stages, list) or not stages or not source.is_file():
                raise ApimartError(f"compiled task {task_id} is malformed")
            checked = []
            for index, stage in enumerate(stages, 1):
                if not isinstance(stage, Mapping) or str(stage.get("stage_id")) != f"S{index}":
                    raise ApimartError(f"compiled task {task_id} has non-sequential stages")
                prompt = str(stage.get(PROMPT_KEY, "")).strip()
                if not prompt:
                    raise ApimartError(f"compiled task {task_id} stage S{index} lacks raw content-only prompt")
                checked.append({"stage_id": f"S{index}", "prompt": prompt})
            return {"task_id": task_id, "source_video": source, "stages": checked}
    raise ApimartError(f"task_id {task_id} is absent from compiled jobs")


def public_url(base_url: str, filename: str) -> str:
    return base_url.rstrip("/") + "/" + urllib.parse.quote(filename)


def upload_reference_image(apimart: ApimartClient, image: Path) -> dict[str, Any]:
    """Retry transport-only image uploads without repeating the image edit."""

    last_error: ApimartError | None = None
    for attempt in range(1, 4):
        try:
            uploaded = apimart.upload_image(image)
            url = uploaded.get("url")
            if isinstance(url, str):
                parsed = urllib.parse.urlsplit(url)
                # APIMart may preserve spaces from the local filename in the
                # returned CDN URL. Encode the path before passing it upstream.
                uploaded["url"] = urllib.parse.urlunsplit((
                    parsed.scheme,
                    parsed.netloc,
                    urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%:@-._~!$&'()*+,;="),
                    parsed.query,
                    parsed.fragment,
                ))
            return uploaded
        except ApimartError as error:
            last_error = error
            if attempt == 3:
                break
            print(json.dumps({
                "event": "reference_image_upload_retry",
                "attempt": attempt,
                "error": str(error),
            }, ensure_ascii=False), flush=True)
            time.sleep(attempt * 3)
    raise ApimartError("reference image upload failed after 3 attempts") from last_error


def relocated_artifact(path_value: Any, relocated_dir: Path) -> Path | None:
    """Resolve an artifact after its bridge directory has been archived."""

    if not isinstance(path_value, str) or not path_value.strip():
        return None
    original = Path(path_value)
    for candidate in (original, relocated_dir / original.name):
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def prior_primary_reference(
    stage_dir: Path,
    previous_video: Path,
    next_prompt: str,
    refiner_model: str,
) -> tuple[dict[str, Any], Path] | None:
    """Find the latest edited primary frame moved under an archived attempt."""

    attempts_root = stage_dir / "attempts"
    attempt_dirs = sorted(
        (
            item for item in attempts_root.iterdir()
            if item.is_dir() and attempt_number_from_path(item) is not None
        ),
        key=lambda item: attempt_number_from_path(item) or 0,
        reverse=True,
    ) if attempts_root.is_dir() else []
    for attempt_dir in attempt_dirs:
        archived_bridge_dir = attempt_dir / "bridge_for_next"
        archived_bridge_path = archived_bridge_dir / "bridge.json"
        if not archived_bridge_path.is_file():
            continue
        bridge = read_json(archived_bridge_path)
        reference_plan = bridge.get("reference_plan", bridge.get("refiner"))
        if not (
            bridge.get("kind") == BRIDGE_KIND
            and bridge.get("stage_id", stage_dir.name) == stage_dir.name
            and bridge.get("previous_video") == str(previous_video)
            and bridge.get("next_raw_prompt") == next_prompt
            and isinstance(reference_plan, Mapping)
            and reference_plan.get("model") == refiner_model
        ):
            continue
        try:
            selected_frame_index = int(
                bridge.get("selected_frame_index", reference_plan.get("selected_frame_index"))
            )
        except (TypeError, ValueError) as error:
            raise ApimartError(f"{attempt_dir.name} bridge lacks Qwen's selected frame index") from error
        if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
            # Archives created before the first-frame style-master contract
            # are not safe to reuse: their middle-frame image would become
            # the style source for the new temporal anchors.
            continue
        reference_images = bridge.get("reference_images")
        if not isinstance(reference_images, list) or not reference_images:
            continue
        # A retry may be recovering either a normal one-anchor bridge or a
        # previous three-anchor bridge (for example when the configured
        # global-style policy was already three anchors).  In both cases the
        # edited primary anchor is the only safe style master to carry into a
        # new attempt.  Prefer its explicit temporal role and fall back to the
        # sole image for legacy one-anchor archives.
        primary_index: int | None = None
        roles = bridge.get("reference_roles")
        if isinstance(roles, list):
            for index, role in enumerate(roles):
                if (
                    isinstance(role, Mapping)
                    and role.get("source_frame_index") == PRIMARY_REFERENCE_FRAME_INDEX
                    and role.get("role") in {"edited start anchor", "edited primary anchor"}
                ):
                    primary_index = index
                    break
        if primary_index is None and len(reference_images) == 1:
            primary_index = 0
        if primary_index is None or primary_index >= len(reference_images):
            continue
        primary_reference = relocated_artifact(reference_images[primary_index], archived_bridge_dir)
        if primary_reference is None:
            raise ApimartError(f"{attempt_dir.name} primary reference image is missing after archival")
        return dict(reference_plan), primary_reference
    return None


def upload_bridge_image(
    apimart: ApimartClient,
    image: Path,
    media_public_base_url: str | None,
    media_dir: Path | None,
) -> dict[str, Any]:
    """Expose one generated reference through the provider's supported route."""

    if apimart.is_ctmoai:
        return {
            "url": apimart.upload_media(image, "images"),
            "upload_mode": "ctmoai_sd_media",
        }
    if media_public_base_url and media_dir is not None:
        media_dir.mkdir(parents=True, exist_ok=True)
        media_image = media_dir / image.name
        if image.resolve() != media_image.resolve():
            shutil.copy2(image, media_image)
        return {
            "url": public_url(media_public_base_url, media_image.name),
            "filename": media_image.name,
            "bytes": media_image.stat().st_size,
            "upload_mode": "local_public_media_server",
        }
    return upload_reference_image(apimart, image)


def temporal_anchor_h3_prompt(
    raw_prompt: str,
    reference_roles: Sequence[Mapping[str, Any]],
) -> str:
    """Build the stable prompt contract for a three-image appearance edit."""

    requirement = normalized_prompt(raw_prompt)
    if not requirement:
        raise ApimartError("three-anchor H3 prompt requires a non-empty atomic requirement")
    if len(reference_roles) != GLOBAL_STYLE_REFERENCE_COUNT:
        raise ApimartError(
            "three-anchor H3 prompt requires exactly "
            f"{GLOBAL_STYLE_REFERENCE_COUNT} temporal roles"
        )
    anchor_lines: list[str] = []
    for picture_index, role in enumerate(reference_roles, 1):
        role_name = normalized_prompt(str(role.get("role", "")))
        source_frame = role.get("source_frame_index")
        if not role_name or not isinstance(source_frame, int):
            raise ApimartError(f"invalid temporal role for Picture {picture_index}")
        anchor_lines.append(
            f"Use <Picture {picture_index}> as the {role_name} at source frame {source_frame}."
        )
    prompt = (
        f"{requirement} {' '.join(anchor_lines)} "
        "Frame-lock the edited appearance to the corresponding temporal anchors: match the start appearance to "
        "<Picture 1>, the primary appearance at its source frame to <Picture 2>, and the end appearance to "
        "<Picture 3>. Maintain this edited appearance in every frame and never revert to the source appearance. "
        "Preserve the scene from <Video 1>; use <Video 1> for motion and temporal progression. "
        "Require a smooth transition in appearance from the start anchor through the primary anchor to the end anchor."
    )
    return validate_h3_reference_tags(prompt, GLOBAL_STYLE_REFERENCE_COUNT, reference_roles)


def three_anchor_reference_plan(
    model: str,
    raw_prompt: str,
    selected_frame_index: int,
    is_global_style: bool,
) -> dict[str, Any]:
    """Describe deterministic temporal-anchor edits without another model call."""

    requirement = normalized_prompt(raw_prompt)
    if not requirement:
        raise ApimartError("three-anchor reference plan requires a non-empty atomic requirement")
    if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
        raise ApimartError(
            "three-anchor reference plan requires the first parent frame as its style master: "
            f"{selected_frame_index} != {PRIMARY_REFERENCE_FRAME_INDEX}"
        )
    return {
        "model": model,
        "selected_frame_index": selected_frame_index,
        "style_reference_frame_index": PRIMARY_REFERENCE_FRAME_INDEX,
        "middle_frame_index": TEMPORAL_MIDDLE_FRAME_INDEX,
        "end_frame_index": TEMPORAL_END_FRAME_INDEX,
        "middle_image_edit_prompt": requirement,
        "end_image_edit_prompt": requirement,
        "image_edit_prompt_source": "raw_atomic_prompt_verbatim",
        "is_global_style": is_global_style,
        "frame_observation": (
            "the first-frame primary is the shared style master; middle and end anchors "
            "are edited from their own parent frames"
        ),
        "usage": {},
    }


def deterministic_repair_h3_prompt(
    raw_prompt: str,
    picture_count: int,
    reference_roles: Sequence[Mapping[str, Any]],
    repair_context: Mapping[str, Any],
) -> str:
    """Build a repair prompt without another free-form Qwen rewrite.

    The current atomic requirement remains the sole semantic source.  The
    repair component contributes one allow-listed clause, while this helper
    reconstructs the exact media-tag contract for the selected references.
    """

    if picture_count not in {0, 1, 3}:
        raise ApimartError(f"unsupported repair reference count: {picture_count}")
    if picture_count == 0:
        base = f"Apply only this edit to <Video 1>: {raw_prompt.strip()}"
    elif picture_count == 1:
        base = (
            "Use <Picture 1> as the visual appearance target throughout the video. "
            "Preserve the scene from <Video 1>; use <Video 1> for motion and temporal progression. "
            f"Apply only this edit to <Video 1>: {raw_prompt.strip()}"
        )
    else:
        if len(reference_roles) != 3:
            raise ApimartError("three-anchor repair requires exactly three reference roles")
        base = temporal_anchor_h3_prompt(raw_prompt, reference_roles)
    try:
        repaired = apply_repair_clause(
            base,
            current_requirement=raw_prompt,
            repair=repair_context,
            picture_count=picture_count,
        )
        return validate_h3_reference_tags(repaired, picture_count, reference_roles)
    except RepairValidationError as error:
        raise ApimartError(str(error)) from error


def bridge_for_stage(
    refiner: DashScopeVisionRefiner,
    editor: GrsaiImageEditor,
    apimart: ApimartClient,
    previous_video: Path,
    stage_dir: Path,
    next_prompt: str,
    global_style_reference_count: int = DEFAULT_STATIC_REFERENCE_COUNT,
    media_public_base_url: str | None = None,
    media_dir: Path | None = None,
    task_id: str | None = None,
    failure_observation: str | None = None,
    repair_context: Mapping[str, Any] | None = None,
) -> tuple[list[str], str, dict[str, Any]]:
    base_policy = reference_policy(next_prompt, global_style_reference_count)
    repair_context = dict(repair_context) if repair_context is not None else None
    if repair_context is not None:
        try:
            repair_context = validate_repair_record(
                repair_context,
                stage_id=stage_dir.name,
                current_requirement=next_prompt,
            )
        except RepairValidationError as error:
            raise ApimartError(str(error)) from error
        policy = dict(base_policy)
        policy["needs_reference_image"] = int(repair_context["reference_image_count"]) > 0
        policy["reference_image_count"] = int(repair_context["reference_image_count"])
        policy["policy_reason"] = f"vetra_repair:{repair_context['repair_action']}"
    else:
        policy = base_policy
    failure_observation = normalized_prompt(failure_observation or "") or None
    bridge_dir = stage_dir / "bridge_for_next"
    bridge_path = bridge_dir / "bridge.json"
    stage_label = stage_dir.name
    file_prefix = f"task_{task_id or 'unknown'}_{stage_label}"
    if bridge_path.is_file():
        bridge = read_json(bridge_path)
        image_urls = bridge.get("image_urls")
        h3_prompt = bridge.get("h3_prompt")
        refiner_metadata = bridge.get("refiner")
        final_refiner_metadata = bridge.get("final_refiner")
        expected_count = int(policy["reference_image_count"])
        saved_selected_frame = bridge.get("selected_frame_index")
        first_frame_reference = (
            expected_count == 0
            or saved_selected_frame == PRIMARY_REFERENCE_FRAME_INDEX
        )
        role_contract_matches = (
            expected_count == 0
            or reference_roles_match_policy(bridge.get("reference_roles"), expected_count)
        )
        if (
            bridge.get("kind") == BRIDGE_KIND
            and bridge.get("stage_id", stage_label) == stage_label
            and bridge.get("next_raw_prompt") == next_prompt
            and bridge.get("previous_video") == str(previous_video)
            and bridge.get("policy") == policy
            and isinstance(h3_prompt, str)
            and isinstance(refiner_metadata, Mapping)
            and refiner_metadata.get("model") == refiner.model
            and isinstance(final_refiner_metadata, Mapping)
            and final_refiner_metadata.get("model") == refiner.model
            and bridge.get("failure_observation") == failure_observation
            and bridge.get("repair_context") == repair_context
            and first_frame_reference
            and role_contract_matches
        ):
            if not isinstance(image_urls, list) or not all(isinstance(url, str) for url in image_urls):
                raise ApimartError("reusable bridge has invalid reference image URLs")
            if len(image_urls) > expected_count:
                raise ApimartError("reusable bridge has more image URLs than its reference policy")
            saved_reference_roles = bridge.get("reference_roles", ())
            if not isinstance(saved_reference_roles, list):
                saved_reference_roles = ()
            validate_h3_reference_tags(h3_prompt, expected_count, saved_reference_roles)
            # Refresh provider-facing URLs from durable local images. This also
            # resumes a bridge interrupted between image editing and upload.
            if apimart.is_ctmoai:
                local_images = bridge.get("reference_images")
                if (
                    len(image_urls) == expected_count
                    and all(
                        url.startswith(apimart.base_url.rstrip("/") + "/sd-media/")
                        for url in image_urls
                    )
                ):
                    # CTMOAI sd-media objects are durable. Re-uploading them
                    # on every resume changes the H3 request fingerprint and
                    # can create a second paid task, so keep the saved URLs.
                    return list(image_urls), h3_prompt, bridge
                if isinstance(local_images, list) and all(
                    isinstance(item, str) and Path(item).is_file() for item in local_images
                ) and len(local_images) == expected_count:
                    refreshed_urls = [apimart.upload_media(Path(item), "images") for item in local_images]
                    bridge["image_urls"] = refreshed_urls
                    bridge["image_uploads"] = [
                        {"url": url, "upload_mode": "ctmoai_sd_media"} for url in refreshed_urls
                    ]
                    write_json(bridge_path, bridge)
                    return refreshed_urls, h3_prompt, bridge
                bridge_path.unlink(missing_ok=True)
            elif media_public_base_url and (
                len(image_urls) != expected_count or any(
                    not url.startswith(media_public_base_url.rstrip("/") + "/") for url in image_urls
                )
            ):
                local_images = bridge.get("reference_images")
                if media_dir is not None and isinstance(local_images, list) and all(
                    isinstance(item, str) and Path(item).is_file() for item in local_images
                ) and len(local_images) == expected_count:
                    refreshed_urls: list[str] = []
                    refreshed_uploads: list[dict[str, Any]] = []
                    for index, item in enumerate(local_images, 1):
                        source_image = Path(item)
                        media_name = f"{file_prefix}_reference_image_{index}.png"
                        media_image = media_dir / media_name
                        shutil.copy2(source_image, media_image)
                        refreshed_urls.append(public_url(media_public_base_url, media_name))
                        refreshed_uploads.append({
                            "url": refreshed_urls[-1], "filename": media_name,
                            "bytes": media_image.stat().st_size,
                            "upload_mode": "local_public_media_server",
                        })
                    bridge["image_urls"] = refreshed_urls
                    bridge["image_uploads"] = refreshed_uploads
                    write_json(bridge_path, bridge)
                    return refreshed_urls, h3_prompt, bridge
                bridge_path.unlink(missing_ok=True)
            else:
                if len(image_urls) != expected_count:
                    raise ApimartError(
                        "reusable bridge reference count does not match its policy: "
                        f"{len(image_urls)} != {expected_count}"
                    )
                return list(image_urls), h3_prompt, bridge
    bridge_dir.mkdir(parents=True, exist_ok=True)
    context_frames = [
        bridge_dir / f"{file_prefix}_context_frame_{frame_index:03d}.png"
        for frame_index in QWEN_CONTEXT_FRAME_INDICES
    ]
    for context_frame, frame_index in zip(context_frames, QWEN_CONTEXT_FRAME_INDICES, strict=True):
        select_keyframe(previous_video, context_frame, frame_index)
    context_by_index = dict(zip(QWEN_CONTEXT_FRAME_INDICES, context_frames, strict=True))
    reference_count = int(policy["reference_image_count"])
    reference_plan: dict[str, Any] = {"model": refiner.model, "mode": "no_reference_image"}
    three_anchor_plan: dict[str, Any] | None = None
    reference_images: list[Path] = []
    image_edits: list[dict[str, Any]] = []
    reference_roles: list[dict[str, Any]] = []

    if bool(policy["needs_reference_image"]):
        prior_reference = (
            prior_primary_reference(stage_dir, previous_video, next_prompt, refiner.model)
            if (
                reference_count == 3
                or (
                    repair_context is not None
                    and bool(repair_context.get("reuse_primary_reference"))
                    and reference_count in {1, 3}
                )
            ) else None
        )
        if prior_reference is not None:
            reference_plan, archived_primary = prior_reference
            selected_frame_index = int(reference_plan["selected_frame_index"])
            primary_reference = bridge_dir / f"{file_prefix}_reference_frame_{selected_frame_index:03d}.png"
            shutil.copy2(archived_primary, primary_reference)
            primary_edit_state = {
                "status": f"reused_from_{archived_primary.parent.parent.name}",
                "selected_frame_index": selected_frame_index,
                "archived_output": str(archived_primary),
                "output": str(primary_reference),
            }
        else:
            reference_plan_path = bridge_dir / f"{file_prefix}_qwen_reference_plan.json"
            reference_plan_state = read_json(reference_plan_path) if reference_plan_path.is_file() else {}
            if (
                reference_plan_state.get("kind") == "qwen_multiframe_reference_plan_v1"
                and reference_plan_state.get("previous_video") == str(previous_video)
                and reference_plan_state.get("raw_prompt") == next_prompt
                and reference_plan_state.get("model") == refiner.model
                and reference_plan_state.get("is_global_style") == bool(policy.get("is_global_style"))
                and isinstance(reference_plan_state.get("result"), Mapping)
                and reference_plan_state["result"].get("selected_frame_index") == PRIMARY_REFERENCE_FRAME_INDEX
            ):
                reference_plan = dict(reference_plan_state["result"])
            else:
                reference_plan = refiner.plan_reference(
                    context_frames,
                    next_prompt,
                    bool(policy.get("is_global_style")),
                )
                write_json(reference_plan_path, {
                    "kind": "qwen_multiframe_reference_plan_v1",
                    "previous_video": str(previous_video),
                    "raw_prompt": next_prompt,
                    "model": refiner.model,
                    "is_global_style": bool(policy.get("is_global_style")),
                    "result": reference_plan,
                })
            # A cached plan may have been produced by an older runner that
            # let Qwen paraphrase the image instruction.  Normalize it here
            # as well as in plan_reference so resume cannot reintroduce extra
            # edits into the image-editor request.
            reference_plan["image_edit_prompt"] = next_prompt.strip()
            reference_plan["image_edit_prompt_source"] = "raw_atomic_prompt_verbatim"
            selected_frame_index = int(reference_plan["selected_frame_index"])
            if selected_frame_index != PRIMARY_REFERENCE_FRAME_INDEX:
                raise ApimartError(
                    "reference plan must use the first parent frame as the primary style master: "
                    f"{selected_frame_index} != {PRIMARY_REFERENCE_FRAME_INDEX}"
                )
            primary_reference = bridge_dir / f"{file_prefix}_reference_frame_{selected_frame_index:03d}.png"
            primary_edit_state = editor.edit(
                context_by_index[selected_frame_index],
                next_prompt,
                str(reference_plan["image_edit_prompt"]),
                primary_reference,
                bridge_dir / f"{file_prefix}_image_edit_state_frame_{selected_frame_index:03d}.json",
            )

        if reference_count == 1:
            reference_images = [primary_reference]
            image_edits = [primary_edit_state]
            reference_roles = expected_reference_roles(reference_count)
        elif reference_count == GLOBAL_STYLE_REFERENCE_COUNT:
            three_anchor_plan_path = bridge_dir / f"{file_prefix}_three_anchor_plan.json"
            three_anchor_plan_state = read_json(three_anchor_plan_path) if three_anchor_plan_path.is_file() else {}
            if (
                three_anchor_plan_state.get("kind") == "three_anchor_reference_plan_v1"
                and three_anchor_plan_state.get("previous_video") == str(previous_video)
                and three_anchor_plan_state.get("raw_prompt") == next_prompt
                and three_anchor_plan_state.get("model") == refiner.model
                and three_anchor_plan_state.get("selected_frame_index") == selected_frame_index
                and three_anchor_plan_state.get("primary_reference") == str(primary_reference)
                and isinstance(three_anchor_plan_state.get("result"), Mapping)
                and three_anchor_plan_state["result"].get("style_reference_frame_index")
                == PRIMARY_REFERENCE_FRAME_INDEX
                and three_anchor_plan_state["result"].get("middle_frame_index")
                == TEMPORAL_MIDDLE_FRAME_INDEX
                and three_anchor_plan_state["result"].get("end_frame_index")
                == TEMPORAL_END_FRAME_INDEX
                and isinstance(three_anchor_plan_state["result"].get("middle_image_edit_prompt"), str)
                and isinstance(three_anchor_plan_state["result"].get("end_image_edit_prompt"), str)
            ):
                three_anchor_plan = dict(three_anchor_plan_state["result"])
            else:
                three_anchor_plan = three_anchor_reference_plan(
                    refiner.model,
                    next_prompt,
                    selected_frame_index,
                    bool(policy.get("is_global_style")),
                )
                write_json(three_anchor_plan_path, {
                    "kind": "three_anchor_reference_plan_v1",
                    "previous_video": str(previous_video),
                    "raw_prompt": next_prompt,
                    "model": refiner.model,
                    "selected_frame_index": selected_frame_index,
                    "primary_reference": str(primary_reference),
                    "result": three_anchor_plan,
                })
            three_anchor_plan["middle_image_edit_prompt"] = next_prompt.strip()
            three_anchor_plan["end_image_edit_prompt"] = next_prompt.strip()
            three_anchor_plan["image_edit_prompt_source"] = "raw_atomic_prompt_verbatim"
            # The first-frame primary is the shared style master. The middle
            # and end anchors are edited from their own parent frames while
            # receiving that first-frame image as an explicit style reference.
            middle_frame_index = TEMPORAL_MIDDLE_FRAME_INDEX
            end_frame_index = TEMPORAL_END_FRAME_INDEX
            middle_reference = bridge_dir / f"{file_prefix}_reference_frame_{middle_frame_index:03d}.png"
            end_reference = bridge_dir / f"{file_prefix}_reference_frame_{end_frame_index:03d}.png"
            middle_edit_state = editor.edit(
                context_by_index[middle_frame_index],
                next_prompt,
                str(three_anchor_plan["middle_image_edit_prompt"]),
                middle_reference,
                bridge_dir / f"{file_prefix}_image_edit_state_frame_{middle_frame_index:03d}.json",
                style_reference=primary_reference,
            )
            end_edit_state = editor.edit(
                context_by_index[end_frame_index],
                next_prompt,
                str(three_anchor_plan["end_image_edit_prompt"]),
                end_reference,
                bridge_dir / f"{file_prefix}_image_edit_state_frame_{end_frame_index:03d}.json",
                style_reference=primary_reference,
            )
            # Picture numbering is semantic and stable: first-frame style
            # master, middle temporal anchor, then end temporal anchor.
            reference_images = [primary_reference, middle_reference, end_reference]
            image_edits = [primary_edit_state, middle_edit_state, end_edit_state]
            reference_roles = expected_reference_roles(reference_count)
        else:
            raise ApimartError(f"unsupported reference image count: {reference_count}")

    if repair_context is not None and repair_context.get("mode") != "fixed_three_anchor":
        h3_prompt = deterministic_repair_h3_prompt(
            next_prompt,
            len(reference_images),
            reference_roles,
            repair_context,
        )
        final_refinement = {
            "model": refiner.model,
            "h3_prompt": h3_prompt,
            "h3_prompt_source": "vetra_deterministic_repair",
            "frame_observation": "repair prompt built from closed-set action",
            "picture_count": len(reference_images),
            "is_global_style": bool(policy.get("is_global_style")),
            "repair_action": repair_context["repair_action"],
            "usage": {},
        }
    else:
        final_refinement = refiner.compose_h3_prompt(
            context_frames,
            reference_images,
            next_prompt,
            bool(policy.get("is_global_style")),
            reference_roles,
            failure_observation,
        )
        h3_prompt = str(final_refinement["h3_prompt"])
    bridge: dict[str, Any] = {
        "kind": BRIDGE_KIND,
        "stage_id": stage_label,
        "previous_video": str(previous_video),
        "next_raw_prompt": next_prompt,
        "policy": policy,
        "context_frame_indices": list(QWEN_CONTEXT_FRAME_INDICES),
        "context_keyframes": [str(frame) for frame in context_frames],
        "selected_frame_index": reference_plan.get("selected_frame_index"),
        "reference_roles": reference_roles,
        "failure_observation": failure_observation,
        "repair_context": repair_context,
        "selection_reason": reference_plan.get("selection_reason", ""),
        "h3_prompt": h3_prompt,
        "refiner": reference_plan,
        "reference_plan": reference_plan,
        "three_anchor_plan": three_anchor_plan,
        "final_refiner": final_refinement,
        "reference_images": [str(image) for image in reference_images],
        "image_edits": image_edits,
        "image_uploads": [],
        "image_urls": [],
    }
    # Persist Qwen and image-edit outputs before network upload so interruption
    # recovery never needs to repeat the paid image edits.
    write_json(bridge_path, bridge)
    for reference_image in reference_images:
        upload = upload_bridge_image(
            apimart,
            reference_image,
            media_public_base_url,
            media_dir,
        )
        bridge["image_uploads"].append({
            key: upload.get(key)
            for key in ("url", "bytes", "created_at", "filename", "upload_mode")
        })
        bridge["image_urls"].append(str(upload["url"]))
        write_json(bridge_path, bridge)
    write_json(bridge_path, bridge)
    return list(bridge["image_urls"]), h3_prompt, bridge
