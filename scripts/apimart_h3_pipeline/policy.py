"""Reference policy and deterministic prompt/reference contracts."""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Mapping, Sequence

try:
    from run_apimart_minimax_h3 import ApimartError
except ModuleNotFoundError:
    from ..run_apimart_minimax_h3 import ApimartError

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
    static_visual = bool(re.search(
        r"\b(recolor|colour|color|style|appearance|look|weather|rain|snow|wet|reflective|"
        r"lighting|light(?:ing)?|shadow|clothing|outfit|dress|background|foreground|scene|face|hair|"
        r"makeup|material|surface|sandwich|napkin|birdhouse|donut|milkshake|newspaper|painting|sepia|"
        r"vintage|particles|glow|depth of field|bokeh|action from .* to)\b",
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


def validate_refined_text(value: str, field: str) -> str:
    value = normalized_prompt(value)
    if not value:
        raise ApimartError(f"Qwen-VL returned an empty {field}")
    if len(value) > 1200:
        raise ApimartError(f"Qwen-VL returned an overlong {field}")
    forbidden = ("subject_definitions:", "retention_analysis:", "detailed_description:", "overall_soundscape:")
    if any(marker in value.lower() for marker in forbidden):
        raise ApimartError(f"Qwen-VL leaked a structured task wrapper into {field}")
    return value


def validate_h3_reference_tags(
    value: str,
    picture_count: int,
    reference_roles: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Validate Qwen's reference/temporal contract without rewriting its prompt."""

    value = validate_refined_text(value, "h3_prompt")
    if "<video 1>" not in value.lower():
        raise ApimartError("Qwen-VL h3_prompt omitted the required <Video 1> tag")
    picture_tags = {int(item) for item in re.findall(r"<picture\s+(\d+)>", value, flags=re.IGNORECASE)}
    expected_tags = set(range(1, picture_count + 1))
    if picture_tags != expected_tags:
        raise ApimartError(
            "Qwen-VL h3_prompt picture tags do not match generated references: "
            f"found={sorted(picture_tags)} expected={sorted(expected_tags)}"
        )
    # The explicit source-frame declaration is required for the three-anchor
    # fallback. A normal one-picture bridge only needs the visual-reference
    # tag, so older concise one-picture prompts remain valid.
    if reference_roles and picture_count == 3:
        if len(reference_roles) != picture_count:
            raise ApimartError(
                "Qwen-VL temporal role count does not match generated references: "
                f"{len(reference_roles)} != {picture_count}"
            )
        lowered = value.lower()
        for picture_index, role in enumerate(reference_roles, 1):
            role_name = normalized_prompt(str(role.get("role", ""))).lower()
            source_frame = role.get("source_frame_index")
            if not role_name or not isinstance(source_frame, int):
                raise ApimartError(f"invalid temporal role for Picture {picture_index}")
            if f"<picture {picture_index}>" not in lowered:
                raise ApimartError(f"h3_prompt omitted temporal Picture {picture_index} anchor")
            # Requiring the source-frame number keeps the anchor mapping in
            # the actual H3 prompt instead of relying on out-of-band metadata.
            if f"source frame {source_frame}" not in lowered:
                raise ApimartError(
                    f"h3_prompt omitted source frame {source_frame} for Picture {picture_index}"
                )
            if role_name not in lowered:
                raise ApimartError(
                    f"h3_prompt omitted role '{role_name}' for Picture {picture_index}"
                )
    return value


def validate_image_edit_prompt(value: str) -> str:
    """Accept Qwen's image-edit instruction verbatim after structural checks."""

    return validate_refined_text(value, "image_edit_prompt")


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


def reference_roles_match_policy(value: Any, reference_count: int) -> bool:
    """Check that a cached bridge uses the current frame/role contract."""

    if not isinstance(value, list):
        return False
    return value == expected_reference_roles(reference_count)

