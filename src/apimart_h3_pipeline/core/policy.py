"""Reference policy and deterministic prompt/reference contracts."""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Mapping, Sequence

from ..providers.apimart import ApimartError

from .constants import (
    DEFAULT_STATIC_REFERENCE_COUNT, GLOBAL_STYLE_REFERENCE_COUNT, REFERENCE_IMAGE_COUNTS,
    PRIMARY_REFERENCE_FRAME_INDEX, TEMPORAL_END_FRAME_INDEX, TEMPORAL_MIDDLE_FRAME_INDEX,
)

def is_global_style_edit(prompt: str) -> bool:
    """Detect edits whose target is the appearance of the complete sequence."""

    text = re.sub(r"\s+", " ", prompt.strip().lower())
    return bool(re.search(
        r"\b(?:visual\s+style|style|stylized|styli[sz]e|appearance|look|film\s+(?:look|effect)|"
        r"sepia|vintage|monochrome|black[ -]?and[ -]?white|watercolou?r|oil\s+paint(?:ing)?|"
        r"sketch|cartoon|anime|color\s+grading|colour\s+grading)\b",
        text,
    ))


def camera_motion_kind(prompt: str) -> str | None:
    """Classify an explicit camera movement without matching camera objects."""

    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if not text:
        return None
    if re.search(r"\bpush[- ]?(?:in|forward)\b", text):
        return "push_in"
    if re.search(r"\bpull[- ]?(?:out|back)\b", text):
        return "pull_out"
    if re.search(r"\b(?:zoom|zooming)\b", text):
        return "zoom"
    if re.search(r"\bdolly(?:ing)?\b", text):
        return "dolly"
    if re.search(r"\btilt(?:ing)?\b", text) and re.search(
        r"\b(?:camera|shot|view|frame|up|down)\b", text,
    ):
        return "tilt"
    if re.search(r"\borbit(?:ing)?\b", text):
        return "orbit"
    if re.search(r"\b(?:tracking|track)\s+shot\b", text):
        return "tracking"
    if re.search(r"\b(?:camera|shot|view)\b.*\btrack(?:ing)?\b|\btrack(?:ing)?\b.*\b(?:camera|shot|view)\b", text):
        return "tracking"
    if re.search(r"\bpan(?:ning)?\b", text) and re.search(
        r"\b(?:camera|shot|view|frame|left|right|up|down|across|over|around|toward|towards)\b", text,
    ):
        return "pan"
    camera_term = r"\b(?:camera|viewpoint|perspective)\b"
    movement_term = r"\b(?:move|moves|moving|movement|motion|translate|translation|shift|slide|travel|track|pan)\w*\b"
    if re.search(rf"{camera_term}.*{movement_term}|{movement_term}.*{camera_term}", text):
        if re.search(r"\b(?:left|right|horizontal|sideways|lateral)\b", text):
            return "pan"
        return "generic"
    return None


def is_camera_motion_edit(prompt: str) -> bool:
    """Return whether an atomic requirement explicitly changes the camera."""

    return camera_motion_kind(prompt) is not None


_DYNAMIC_ACTION_RE = re.compile(
    r"\b(?:action|actions|pose|poses|posture|postures|gesture|gestures|"
    r"jump|jumping|run|running|walk|walking|sit|sitting|stand|standing|"
    r"row|rowing|flap|flapping|dance|dancing|swim|swimming|fly|flying|"
    r"turn|turning|wave|waving|crawl|crawling|sway|swaying)\b",
    re.IGNORECASE,
)
_STATIC_ACTION_CONTEXT_RE = re.compile(
    r"\b(?:add|remove|replace|insert|swap|object|prop|newspaper|whiteboard|"
    r"sign|text|digit|number|letter|writing|write|drawing|draw|visible|"
    r"touch|touching|reach|reaching|grab|grabbing|catch|catching|throw|throwing|"
    r"toss|tossing|pick|picking|plant|planting|press|pressing|place|placing|"
    r"remove|removing|lift|lifting|handle|handling|"
    r"pour|pouring|adjust|adjusting|inspect|inspecting|tap|tapping|"
    r"hold|holding|carry|carrying|read|reading|use|using|look|looking|"
    r"position|frame|side|edge|center|central|third|inward|throughout|within|"
    r"balcony|leaf|branch|handhold|cane|fruit|pan|hat|chip|knob|plate|"
    r"sunglasses|binoculars|journal|book|camera|phone|tablet|bottle|"
    r"style|appearance|"
    r"look|color|colour|lighting|background|foreground|material|surface|"
    r"clothing|outfit|face|hair|makeup|wet|rain|snow|glow|particles)\b",
    re.IGNORECASE,
)
_SCENE_TARGET_CONTEXT_RE = re.compile(
    r"\b(?:on|onto|into|near|beside|under|over|through|toward|towards|at)\s+"
    r"(?:a|an|the|their|his|her|its|same|each)\b",
    re.IGNORECASE,
)


def is_dynamic_action_edit(prompt: str) -> bool:
    """Return whether a requirement is a pure temporal action/pose edit."""

    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if not text or is_camera_motion_edit(text) or not _DYNAMIC_ACTION_RE.search(text):
        return False
    return not (_STATIC_ACTION_CONTEXT_RE.search(text) or _SCENE_TARGET_CONTEXT_RE.search(text))


def reference_policy(prompt: str, global_style_reference_count: int = DEFAULT_STATIC_REFERENCE_COUNT) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if global_style_reference_count not in REFERENCE_IMAGE_COUNTS:
        choices = ", ".join(str(item) for item in REFERENCE_IMAGE_COUNTS)
        raise ApimartError(f"static reference count must be one of: {choices}")
    decision_text = re.split(
        r"\b(?:to\s+keep|while\s+(?:keeping|preserving)|keep|preserv\w+|to\s+emphasize)\b",
        text,
        maxsplit=1,
    )[0].strip()
    global_style = is_global_style_edit(decision_text)
    dynamic_action = is_dynamic_action_edit(decision_text)
    static_visual = bool(re.search(
        r"\b(recolor|colour|color|style|appearance|look|weather|rain|snow|wet|reflective|"
        r"lighting|light(?:ing)?|shadow|clothing|outfit|dress|background|foreground|scene|face|hair|"
        r"makeup|material|surface|sandwich|napkin|birdhouse|donut|milkshake|newspaper|painting|sepia|"
        r"vintage|particles|glow|depth of field|bokeh)\b",
        decision_text,
    ))
    generic_edit = bool(re.search(r"\b(add|remove|replace|reposition|move|shift|change the position|transform)\b", decision_text))
    temporal = bool(re.search(
        r"\b(camera|pan|push[- ]?in|pull[- ]?out|zoom|dolly|tilt|orbit|tracking|track(?:ing)? shot|"
        r"shot static|motion|movement|speed|faster|slower|sway|swaying|shake|jitter|blur|temporal|"
        r"frame rate|fps|audio|sound|music|voice|sing(?:ing)?|wind)\b",
        decision_text,
    ))
    if global_style:
        return {
            "needs_reference_image": True,
            "policy_reason": "global_visual_style_change",
            "is_global_style": True,
            # Global style is intentionally one-anchor on the normal
            # attempt.  A failed attempt is escalated explicitly by the
            # stage repair policy, never by this ordinary static override.
            "reference_image_count": DEFAULT_STATIC_REFERENCE_COUNT,
        }
    if dynamic_action:
        return {
            "needs_reference_image": False,
            "policy_reason": "pure_temporal_action_or_pose_change",
            "is_global_style": False,
            "reference_image_count": 0,
        }
    if static_visual:
        return {
            "needs_reference_image": True,
            "policy_reason": "static_visual_or_compositional_change",
            "is_global_style": False,
            "reference_image_count": 3 if global_style_reference_count == 3 else 1,
        }
    if temporal:
        return {
            "needs_reference_image": False,
            "policy_reason": "camera_temporal_or_audio_change",
            "is_global_style": False,
            "reference_image_count": 0,
        }
    if generic_edit:
        return {
            "needs_reference_image": True,
            "policy_reason": "object_or_composition_edit_without_temporal_cue",
            "is_global_style": False,
            "reference_image_count": 3 if global_style_reference_count == 3 else 1,
        }
    return {
        "needs_reference_image": True,
        "policy_reason": "ambiguous_default_to_reference_for_content_fidelity",
        "is_global_style": False,
        "reference_image_count": 3 if global_style_reference_count == 3 else 1,
    }


def no_proxy_opener() -> urllib.request.OpenerDirector:
    """DashScope and GRSAI use direct routes; APIMart inherits the Clash proxy."""

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_json_object(text: str, source: str) -> dict[str, Any]:
    """Parse one fenced or unfenced JSON object returned by a constrained model."""

    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ApimartError(f"{source} returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise ApimartError(f"{source} returned a non-object JSON value")
    return parsed


def normalized_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def expected_reference_roles(reference_count: int) -> list[dict[str, Any]]:
    """Return the stable role contract for the attached reference images."""

    if reference_count == DEFAULT_STATIC_REFERENCE_COUNT:
        return [{
            "picture_index": 1,
            "role": "edited primary anchor",
            "source_frame_index": PRIMARY_REFERENCE_FRAME_INDEX,
        }]
    if reference_count == GLOBAL_STYLE_REFERENCE_COUNT:
        return [
            {
                "picture_index": 1,
                "role": "edited start anchor",
                "source_frame_index": PRIMARY_REFERENCE_FRAME_INDEX,
            },
            {
                "picture_index": 2,
                "role": "edited primary anchor",
                "source_frame_index": TEMPORAL_MIDDLE_FRAME_INDEX,
            },
            {
                "picture_index": 3,
                "role": "edited end anchor",
                "source_frame_index": TEMPORAL_END_FRAME_INDEX,
            },
        ]
    raise ApimartError(f"unsupported reference image count: {reference_count}")
