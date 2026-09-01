"""Failure diagnosis and stage-local repair policy for the VETRA H3 runner.

This module deliberately has no network or media dependencies.  It turns a
structured observer result into a closed-set repair action, builds a minimal
H3 prompt for the retry, and validates the stage-local invariants.  Keeping
these decisions deterministic prevents an observer's free-form text from
silently changing the frozen execution plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FAILURE_TYPES = frozenset(
    {
        "none",
        "edit_missing",
        "identity_drift",
        "previous_stage_lost",
        "style_inconsistency",
        "motion_weak",
        "composition_weak",
        "unclassified",
        "observer_unavailable",
        "not_frame_judgeable",
        "media_invalid",
    }
)

REPAIRABLE_FAILURE_TYPES = frozenset(
    {
        "edit_missing",
        "identity_drift",
        "previous_stage_lost",
        "style_inconsistency",
        "motion_weak",
        "composition_weak",
    }
)

REPAIR_ACTIONS = frozenset(
    {
        "strengthen_edit",
        "strengthen_identity_preservation",
        "strengthen_previous_stage_preservation",
        "use_three_anchor",
        "strengthen_motion",
        "strengthen_composition",
        "no_automatic_repair",
    }
)

REFERENCE_POLICIES = frozenset({"video_only", "one_anchor", "three_anchor"})
# Semantic recovery is deliberately bounded to one retry per stage.  This is
# part of the execution protocol, not a runtime experiment knob.
STAGE_RETRY_LIMIT = 1

_FAILURE_ALIASES = {
    "": "unclassified",
    "none": "none",
    "ok": "none",
    "success": "none",
    "edit_absent": "edit_missing",
    "edit_not_visible": "edit_missing",
    "missing_edit": "edit_missing",
    "edit_missing_or_partial": "edit_missing",
    "identity_failure": "identity_drift",
    "identity_preservation_failure": "identity_drift",
    "previous_edit_lost": "previous_stage_lost",
    "history_lost": "previous_stage_lost",
    "style_drift": "style_inconsistency",
    "style_inconsistent": "style_inconsistency",
    "weak_motion": "motion_weak",
    "camera_motion_weak": "motion_weak",
    "weak_composition": "composition_weak",
    "layout_weak": "composition_weak",
    "observer_error": "observer_unavailable",
    "observer_error_or_unavailable": "observer_unavailable",
    "not_judgeable": "not_frame_judgeable",
    "media_failure": "media_invalid",
}

_ACTION_BY_FAILURE = {
    "edit_missing": "strengthen_edit",
    "identity_drift": "strengthen_identity_preservation",
    "previous_stage_lost": "strengthen_previous_stage_preservation",
    "style_inconsistency": "use_three_anchor",
    "motion_weak": "strengthen_motion",
    "composition_weak": "strengthen_composition",
}

_CLAUSE_BY_ACTION = {
    "strengthen_edit": (
        "Make the requested current edit clearly visible across the sequence. "
        "Apply only this current edit."
    ),
    "strengthen_identity_preservation": (
        "Preserve the identity, count, role, face, body, clothing, and all unedited appearance "
        "of existing people."
    ),
    "strengthen_previous_stage_preservation": (
        "Preserve all edits already confirmed before this stage, then apply only the current edit."
    ),
    "use_three_anchor": (
        "Use the start, primary, and end references only as temporal appearance anchors; "
        "preserve the source video's motion and progression."
    ),
    "strengthen_motion": (
        "Make the requested motion visibly clear with its stated type, direction, amplitude, and speed; "
        "do not introduce a different camera or object motion."
    ),
    "strengthen_composition": (
        "Make the requested spatial or compositional change clearly visible at its stated target position; "
        "preserve all unrequested layout."
    ),
}

_MOTION_RE = re.compile(
    r"\b(camera|pan|push[- ]?in|pull[- ]?out|zoom|dolly|tilt|orbit|tracking|track(?:ing)? shot|"
    r"motion|movement|speed|faster|slower|sway|shake|jitter|temporal|frame rate|fps|audio|sound|music|voice)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "apply",
        "as",
        "at",
        "be",
        "by",
        "change",
        "clearly",
        "edit",
        "for",
        "from",
        "in",
        "into",
        "make",
        "only",
        "preserve",
        "requested",
        "the",
        "this",
        "to",
        "use",
        "with",
    }
)


class RepairValidationError(ValueError):
    """Raised when an observer or repair result violates the stage contract."""


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_failure_type(value: Any, *, success: bool | None, observation: str = "") -> str:
    """Normalize aliases while keeping semantic and transport failures separate."""

    raw = normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    normalized = _FAILURE_ALIASES.get(raw, raw)
    evidence = normalize_text(observation).lower().replace("-", "_")
    if success is True and "not_frame_judgeable" in evidence:
        normalized = "not_frame_judgeable"
    if normalized not in FAILURE_TYPES:
        return "unclassified"
    if success is True and normalized == "unclassified":
        return "none"
    if success is True and normalized in REPAIRABLE_FAILURE_TYPES:
        raise RepairValidationError(
            f"observer marked success=true with semantic failure_type={normalized}"
        )
    if success is False and normalized in {"none", "not_frame_judgeable", "observer_unavailable"}:
        raise RepairValidationError(
            f"observer marked success=false with non-actionable failure_type={normalized}"
        )
    if success is None and normalized == "unclassified":
        return "observer_unavailable"
    return normalized


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def meaningful_tokens(text: str) -> set[str]:
    """Return conservative content tokens used by the no-new-requirement guard."""

    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9']+", normalize_text(text))
    }
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}


def confirmed_stage_ids(previous_requirements: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    for item in previous_requirements:
        if not isinstance(item, Mapping):
            continue
        status = normalize_text(item.get("status", "confirmed")).lower()
        stage_id = normalize_text(item.get("stage_id"))
        if stage_id and status in {"confirmed", "success", "completed"}:
            result.append(stage_id)
    return result


def validate_observation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one observer response for persistence."""

    if not isinstance(result, Mapping):
        raise RepairValidationError("observer result must be a JSON object")
    success = result.get("success")
    if success not in (True, False, None):
        raise RepairValidationError("observer success must be true, false, or null")
    observation = normalize_text(result.get("observation", result.get("observer_evidence", "")))
    failure_type = normalize_failure_type(
        result.get("failure_type"), success=success, observation=observation,
    )
    if success is True and failure_type not in {"none", "not_frame_judgeable"}:
        raise RepairValidationError("successful observer result has an actionable failure type")
    if success is False and failure_type not in REPAIRABLE_FAILURE_TYPES | {"unclassified", "media_invalid"}:
        raise RepairValidationError("failed observer result has an invalid failure type")
    if success is None and failure_type not in {"observer_unavailable", "not_frame_judgeable", "unclassified"}:
        raise RepairValidationError("unavailable observer result has an invalid failure type")
    normalized = dict(result)
    normalized.update({
        "success": success,
        "failure_type": failure_type,
        "observation": observation,
        "observer_evidence": normalize_text(result.get("observer_evidence", observation)),
        "confidence": _confidence(result.get("confidence", 0.0)),
    })
    return normalized


def stage_outcome(observation: Mapping[str, Any], *, allow_unverified: bool = False) -> str:
    """Classify whether a stage may update its parent video."""

    normalized = validate_observation(observation)
    if normalized["success"] is True:
        return "success"
    if normalized["success"] is None:
        return "unverified_success" if allow_unverified else "observation_pending"
    return "semantic_failure"


@dataclass(frozen=True)
class FailureDiagnosisAndRepair:
    """Deterministic stage-local repair policy used by the online runner."""

    confidence_threshold: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise RepairValidationError("confidence_threshold must be between 0 and 1")

    def diagnose(
        self,
        observation: Mapping[str, Any],
        *,
        stage_id: str,
        current_requirement: str,
        failed_prompt: str,
        previous_requirements: Sequence[Mapping[str, Any]] = (),
        attempt: int = 1,
        evidence_frames: Sequence[int] = (),
    ) -> dict[str, Any]:
        if not normalize_text(stage_id):
            raise RepairValidationError("stage_id is required for diagnosis")
        current = normalize_text(current_requirement)
        failed = normalize_text(failed_prompt)
        if not current or not failed:
            raise RepairValidationError("current_requirement and failed_prompt are required")
        if attempt < 1:
            raise RepairValidationError("diagnosis attempt must be positive")
        if any(not isinstance(frame, int) or isinstance(frame, bool) for frame in evidence_frames):
            raise RepairValidationError("evidence_frames must contain integers")
        normalized = validate_observation(observation)
        confidence = float(normalized["confidence"])
        failure_type = str(normalized["failure_type"])
        repairable = (
            normalized["success"] is False
            and failure_type in REPAIRABLE_FAILURE_TYPES
            and confidence >= self.confidence_threshold
        )
        if failure_type == "previous_stage_lost" and not confirmed_stage_ids(previous_requirements):
            repairable = False
        return {
            "kind": "qwen_vl_failure_diagnosis_v1",
            "stage_id": normalize_text(stage_id),
            "attempt": attempt,
            "success": normalized["success"],
            "failure_type": failure_type,
            "observer_evidence": normalized["observer_evidence"],
            "observation": normalized["observation"],
            "confidence": confidence,
            "repairable": repairable,
            "affected_scope": "current_stage_only",
            "preserved_stage_ids": confirmed_stage_ids(previous_requirements),
            "evidence_frames": list(evidence_frames),
            "current_requirement": current,
            "failed_prompt": failed,
        }

    def repair(
        self,
        diagnosis: Mapping[str, Any],
        *,
        stage_id: str,
        current_requirement: str,
        failed_prompt: str,
        previous_requirements: Sequence[Mapping[str, Any]] = (),
        retry_index: int,
        original_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = normalize_text(current_requirement)
        failed = normalize_text(failed_prompt)
        if not current or not failed:
            raise RepairValidationError("current_requirement and failed_prompt are required")
        if normalize_text(diagnosis.get("stage_id")) != normalize_text(stage_id):
            raise RepairValidationError("diagnosis stage_id does not match the current stage")
        if diagnosis.get("affected_scope") != "current_stage_only":
            raise RepairValidationError("repair may only target current_stage_only")
        if diagnosis.get("success") is not False or not diagnosis.get("repairable"):
            raise RepairValidationError("diagnosis is not eligible for semantic repair")
        if retry_index != STAGE_RETRY_LIMIT:
            raise RepairValidationError(
                f"retry_index {retry_index} exceeds the fixed stage repair budget {STAGE_RETRY_LIMIT}"
            )
        failure_type = normalize_failure_type(
            diagnosis.get("failure_type"), success=False,
            observation=str(diagnosis.get("observer_evidence", "")),
        )
        action = _ACTION_BY_FAILURE.get(failure_type, "no_automatic_repair")
        if action == "no_automatic_repair":
            raise RepairValidationError(f"failure type is not repairable: {failure_type}")
        if failure_type == "previous_stage_lost" and not confirmed_stage_ids(previous_requirements):
            raise RepairValidationError("previous_stage_lost requires at least one confirmed parent stage")
        if failure_type == "motion_weak" and not _MOTION_RE.search(current):
            raise RepairValidationError("motion_weak repair requires a motion cue in the current requirement")

        reference_policy, reference_count = self._reference_policy(action, original_policy)
        clause = _CLAUSE_BY_ACTION[action]
        repaired_prompt = self._build_base_prompt(current, failed, clause, reference_count)
        self.validate_repaired_prompt(repaired_prompt, current, action)
        return {
            "kind": "vetra_failure_repair_v1",
            "stage_id": normalize_text(stage_id),
            "retry_index": retry_index,
            "max_retries": STAGE_RETRY_LIMIT,
            "failure_type": failure_type,
            "repair_action": action,
            "repair_clause": clause,
            "repaired_h3_prompt": repaired_prompt,
            "reference_policy": reference_policy,
            "reference_image_count": reference_count,
            "reuse_primary_reference": reference_count in {1, 3},
            "allowed_semantic_source": "current_requirement_only",
            "preservation_source": confirmed_stage_ids(previous_requirements),
            "observer_evidence": normalize_text(diagnosis.get("observer_evidence", "")),
            "guard": {
                "same_stage": True,
                "topology_changed": False,
                "new_requirement_added": False,
                "clt_written": False,
                "ceg_written": False,
            },
        }

    @staticmethod
    def _reference_policy(action: str, original_policy: Mapping[str, Any]) -> tuple[str, int]:
        needs_reference = bool(original_policy.get("needs_reference_image"))
        original_count = int(original_policy.get("reference_image_count", 0) or 0)
        if action == "strengthen_motion":
            return "video_only", 0
        if action == "use_three_anchor":
            return "three_anchor", 3
        if not needs_reference:
            return "video_only", 0
        if original_count == 3:
            return "three_anchor", 3
        return "one_anchor", 1

    @staticmethod
    def _build_base_prompt(current: str, failed: str, clause: str, reference_count: int) -> str:
        # The returned record is a semantic candidate.  The bridge later adds
        # the exact Picture/Video temporal contract for the selected references.
        video_binding = "<Video 1>"
        current_sentence = normalize_text(current).rstrip(" .!?;")
        if reference_count:
            return (
                f"Use the attached visual reference{'' if reference_count == 1 else 's'} for appearance. "
                f"Preserve the source motion from {video_binding}. Apply only this current edit: {current_sentence}. {clause}"
            )
        return f"Apply only this edit to {video_binding}: {current_sentence}. {clause}"

    @staticmethod
    def validate_repaired_prompt(prompt: str, current_requirement: str, action: str) -> str:
        value = normalize_text(prompt)
        current = normalize_text(current_requirement)
        if action not in REPAIR_ACTIONS or action == "no_automatic_repair":
            raise RepairValidationError(f"invalid repair action: {action}")
        if not value:
            raise RepairValidationError("repaired_h3_prompt is empty")
        if len(value) > 1200:
            raise RepairValidationError("repaired_h3_prompt is overlong")
        missing = meaningful_tokens(current) - meaningful_tokens(value)
        if missing:
            raise RepairValidationError(
                "repaired_h3_prompt dropped current requirement tokens: " + ", ".join(sorted(missing))
            )
        lowered = value.lower()
        if "<video 1>" not in lowered:
            raise RepairValidationError("repaired_h3_prompt omitted <Video 1>")
        forbidden = ("subject_definitions:", "retention_analysis:", "detailed_description:", "overall_soundscape:")
        if any(marker in lowered for marker in forbidden):
            raise RepairValidationError("repaired_h3_prompt leaked a structured task wrapper")
        return value


def apply_repair_clause(
    h3_prompt: str,
    *,
    current_requirement: str,
    repair: Mapping[str, Any],
    picture_count: int,
) -> str:
    """Attach a validated repair clause while preserving media tags/contracts."""

    action = normalize_text(repair.get("repair_action"))
    clause = normalize_text(repair.get("repair_clause"))
    if action not in _CLAUSE_BY_ACTION or clause != _CLAUSE_BY_ACTION[action]:
        raise RepairValidationError("repair clause does not match its closed-set action")
    base = normalize_text(h3_prompt)
    current = normalize_text(current_requirement)
    if not base:
        raise RepairValidationError("cannot repair an empty H3 prompt")
    required = meaningful_tokens(current) - meaningful_tokens(base)
    if required:
        base = f"{base} Apply only this current edit: {current}."
    repaired = f"{base} {clause}"
    if picture_count == 0 and "<picture" in repaired.lower():
        raise RepairValidationError("video-only repaired prompt contains a Picture tag")
    return FailureDiagnosisAndRepair.validate_repaired_prompt(repaired, current, action)


def validate_repair_record(
    repair: Mapping[str, Any],
    *,
    stage_id: str,
    current_requirement: str,
) -> dict[str, Any]:
    """Validate a persisted repair record before it controls a paid retry."""

    if not isinstance(repair, Mapping):
        raise RepairValidationError("repair record must be a JSON object")
    if repair.get("kind") != "vetra_failure_repair_v1":
        raise RepairValidationError("unsupported repair record kind")
    if normalize_text(repair.get("stage_id")) != normalize_text(stage_id):
        raise RepairValidationError("repair record stage_id does not match the current stage")
    retry_index = repair.get("retry_index")
    if not isinstance(retry_index, int) or retry_index < 1:
        raise RepairValidationError("repair record retry_index must be a positive integer")
    if retry_index != STAGE_RETRY_LIMIT:
        raise RepairValidationError(
            f"repair record must use the fixed retry index {STAGE_RETRY_LIMIT}"
        )
    persisted_limit = repair.get("max_retries")
    if persisted_limit is not None and (
        not isinstance(persisted_limit, int)
        or isinstance(persisted_limit, bool)
        or persisted_limit != STAGE_RETRY_LIMIT
    ):
        raise RepairValidationError(
            f"repair record must use the fixed retry limit {STAGE_RETRY_LIMIT}"
        )
    action = normalize_text(repair.get("repair_action"))
    if action not in _CLAUSE_BY_ACTION:
        raise RepairValidationError(f"unsupported repair action: {action}")
    reference_policy = normalize_text(repair.get("reference_policy"))
    reference_count = repair.get("reference_image_count")
    if not isinstance(reference_count, int) or isinstance(reference_count, bool):
        raise RepairValidationError("repair reference_image_count must be an integer")
    expected_policy = "video_only" if reference_count == 0 else ("one_anchor" if reference_count == 1 else "three_anchor" if reference_count == 3 else "")
    if reference_policy not in REFERENCE_POLICIES or reference_policy != expected_policy:
        raise RepairValidationError("repair reference policy/count mismatch")
    guard = repair.get("guard")
    guard_keys = ("topology_changed", "new_requirement_added", "clt_written", "ceg_written")
    if not isinstance(guard, Mapping) or any(guard.get(key) is not False for key in guard_keys):
        raise RepairValidationError("repair record violates a control-plane guard")
    if guard.get("same_stage") is not True:
        raise RepairValidationError("repair record is not stage-local")
    prompt = normalize_text(repair.get("repaired_h3_prompt"))
    current = normalize_text(current_requirement)
    FailureDiagnosisAndRepair.validate_repaired_prompt(prompt, current, action)
    clause = normalize_text(repair.get("repair_clause"))
    if clause != _CLAUSE_BY_ACTION[action]:
        raise RepairValidationError("repair record clause does not match its action")
    normalized = dict(repair)
    normalized.update({
        "stage_id": normalize_text(stage_id),
        "retry_index": retry_index,
        "repair_action": action,
        "repair_clause": clause,
        "repaired_h3_prompt": prompt,
        "reference_policy": reference_policy,
        "reference_image_count": int(reference_count),
    })
    return normalized


def fixed_three_anchor_repair(
    *,
    stage_id: str,
    current_requirement: str,
    failed_prompt: str,
    observer_evidence: str,
    retry_index: int = 1,
) -> dict[str, Any]:
    """Represent the legacy all-failures three-anchor fallback explicitly."""

    return _three_anchor_repair(
        stage_id=stage_id,
        current_requirement=current_requirement,
        failed_prompt=failed_prompt,
        observer_evidence=observer_evidence,
        retry_index=retry_index,
        failure_type="unclassified",
        mode="fixed_three_anchor",
    )


def global_style_three_anchor_repair(
    *,
    stage_id: str,
    current_requirement: str,
    failed_prompt: str,
    observer_evidence: str,
    failure_type: str,
    retry_index: int = 1,
) -> dict[str, Any]:
    """Escalate a failed one-anchor global-style edit to temporal anchors.

    The escalation is selected from the already frozen reference policy.  It
    does not reinterpret the requirement or add a new edit; it only changes
    the appearance-conditioning contract for the retry.
    """

    normalized_failure = normalize_failure_type(
        failure_type,
        success=False,
        observation=observer_evidence,
    )
    return _three_anchor_repair(
        stage_id=stage_id,
        current_requirement=current_requirement,
        failed_prompt=failed_prompt,
        observer_evidence=observer_evidence,
        retry_index=retry_index,
        failure_type=normalized_failure,
        mode="global_style_three_anchor",
    )


def _three_anchor_repair(
    *,
    stage_id: str,
    current_requirement: str,
    failed_prompt: str,
    observer_evidence: str,
    retry_index: int,
    failure_type: str,
    mode: str,
) -> dict[str, Any]:
    """Build the shared, validated record for an explicit anchor escalation."""

    if retry_index != STAGE_RETRY_LIMIT:
        raise RepairValidationError(
            f"three-anchor retry must use the fixed retry index {STAGE_RETRY_LIMIT}"
        )
    current = normalize_text(current_requirement)
    failed = normalize_text(failed_prompt)
    if not current or not failed:
        raise RepairValidationError("current_requirement and failed_prompt are required")
    clause = _CLAUSE_BY_ACTION["use_three_anchor"]
    candidate = FailureDiagnosisAndRepair._build_base_prompt(current, failed, clause, 3)
    FailureDiagnosisAndRepair.validate_repaired_prompt(candidate, current, "use_three_anchor")
    return {
        "kind": "vetra_failure_repair_v1",
        "stage_id": normalize_text(stage_id),
        "retry_index": retry_index,
        "max_retries": STAGE_RETRY_LIMIT,
        "failure_type": failure_type,
        "repair_action": "use_three_anchor",
        "repair_clause": clause,
        "repaired_h3_prompt": candidate,
        "reference_policy": "three_anchor",
        "reference_image_count": 3,
        "reuse_primary_reference": True,
        "allowed_semantic_source": "current_requirement_only",
        "preservation_source": [],
        "observer_evidence": normalize_text(observer_evidence),
        "mode": mode,
        "guard": {
            "same_stage": True,
            "topology_changed": False,
            "new_requirement_added": False,
            "clt_written": False,
            "ceg_written": False,
        },
    }
