#!/usr/bin/env python3
"""Split camera-motion and global-style rounds out of compact plans.

The source directory is never modified. Every matching round is removed as a
whole from the copied plans, while its original round number and instruction
are retained in one or both category manifests.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


_GLOBAL_SCOPE_RE = re.compile(
    r"\b(?:entire|whole|overall|throughout|scene|video|frame|visual\s+style|"
    r"visual\s+aesthetic|art(?:istic)?\s+style|style\s+of\s+the\s+(?:scene|video|frame))\b",
    re.IGNORECASE,
)
_EXPLICIT_STYLE_RE = re.compile(
    r"\b(?:visual\s+style|visual\s+aesthetic|art(?:istic)?\s+style|stylization|"
    r"stylisation|stylized|stylised|aesthetic|watercolou?r|oil[- ]painting|"
    r"cartoon|pixar|anime|ghibli|cyberpunk|vintage|sepia|monochrome|"
    r"black[- ]and[- ]white|film\s+noir|sketch|claymation|stop[- ]motion|"
    r"lego|pixel[- ]art|van\s+gogh|painterly|cinematic\s+(?:style|look)|"
    r"film\s+(?:style|look|aesthetic))\b",
    re.IGNORECASE,
)


def is_global_style_round(prompt: str, classifier) -> bool:
    """Use the runtime classifier with scope disambiguation for extraction.

    The runtime predicate intentionally accepts broad appearance/look language
    for reference selection. Extraction is stricter: a round must explicitly
    describe a scene-wide visual style, not a subject expression, lens look, or
    local stylized object.
    """
    if not classifier(prompt):
        return False
    text = " ".join(prompt.split())
    # A black-and-white patterned garment is an object description, not a
    # monochrome visual treatment of the complete sequence.
    if re.search(r"\bblack[- ]and[- ]white\s+pattern(?:ed|ing)?\b", text, re.IGNORECASE):
        return False
    if not _EXPLICIT_STYLE_RE.search(text):
        # A bare "look" or "appearance" is not enough to call a task global.
        return bool(
            _GLOBAL_SCOPE_RE.search(text)
            and re.search(r"\b(?:look|appearance)\b", text, re.IGNORECASE)
            and re.search(r"\b(?:scene|video|frame|entire|whole|overall|throughout)\b", text, re.IGNORECASE)
        )
    # Generic style words ("style", "stylized") need a global target. Named
    # visual media styles may carry their scope in the target noun itself.
    if re.search(r"\b(?:style|stylized|stylised|stylization|stylisation|aesthetic)\b", text, re.IGNORECASE):
        return bool(_GLOBAL_SCOPE_RE.search(text))
    return bool(_GLOBAL_SCOPE_RE.search(text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    src = args.source.resolve()
    out = args.output.resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory does not exist: {src}")
    if out.exists():
        raise SystemExit(f"output directory already exists: {out}")

    # Reuse the pipeline's production classifiers so extraction agrees with
    # runtime camera and global-style reference policies.
    project_src = Path("/root/Pipline/src").resolve()
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from apimart_h3_pipeline.core.policy import (  # noqa: PLC0415
        camera_motion_kind,
        is_global_style_edit,
    )

    plan_out = out / "compact_plans_without_camera_or_global_style"
    plan_out.mkdir(parents=True, exist_ok=True)
    camera_records: list[dict[str, Any]] = []
    style_records: list[dict[str, Any]] = []
    combined_records: list[dict[str, Any]] = []
    task_stats: list[dict[str, Any]] = []

    paths = sorted(src.glob("task_*.json"))
    if not paths:
        raise SystemExit(f"no task JSON files found in {src}")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rounds = data.get("video_diffusion_rounds")
        if not isinstance(rounds, list):
            raise SystemExit(f"{path.name}: video_diffusion_rounds is not a list")
        kept: list[dict[str, Any]] = []
        removed_count = 0
        for original_index, item in enumerate(rounds, 1):
            if not isinstance(item, dict) or not isinstance(item.get("instruction"), str):
                raise SystemExit(f"{path.name}: invalid round {original_index}")
            instruction = item["instruction"]
            camera_kind = camera_motion_kind(instruction)
            style = is_global_style_round(instruction, is_global_style_edit)
            record = {
                "task_id": str(data.get("task_id", "")),
                "source_file": path.name,
                "original_round": int(item.get("round", original_index)),
                "instruction": instruction,
                "camera_motion_kind": camera_kind,
                "global_style": bool(style),
            }
            if camera_kind is not None:
                camera_records.append(record)
            if style:
                style_records.append(record)
            if camera_kind is not None or style:
                combined_records.append({
                    **record,
                    "categories": (["camera_motion"] if camera_kind is not None else [])
                    + (["global_style"] if style else []),
                })
                removed_count += 1
            else:
                kept.append(item)
        # Re-number only the remaining list; extraction records retain original
        # round numbers, so no information about the split is lost.
        for new_index, item in enumerate(kept, 1):
            item["round"] = new_index
        output_data = dict(data)
        output_data["video_diffusion_rounds"] = kept
        (plan_out / path.name).write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        task_stats.append({
            "task_id": str(data.get("task_id", "")),
            "source_file": path.name,
            "original_round_count": len(rounds),
            "remaining_round_count": len(kept),
            "removed_round_count": removed_count,
        })

    # Retain the original bundle's repair manifest as a reference artifact.
    source_manifest = src.parent / "compact_plans_qwen38_flash_applied_repairs.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, out / source_manifest.name)
    for name, records in (
        ("camera_motion_rounds.json", camera_records),
        ("global_style_rounds.json", style_records),
        ("camera_or_global_style_rounds.json", combined_records),
    ):
        (out / name).write_text(
            json.dumps({"records": records, "record_count": len(records)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "source": str(src),
        "output": str(out),
        "classifier": "apimart_h3_pipeline.core.policy",
        "rule": "remove an entire round when it contains camera motion or global visual style",
        "rounds_are_renumbered_after_removal": True,
        "original_round_numbers_preserved_in_extracted_records": True,
        "task_count": len(paths),
        "camera_motion_round_count": len(camera_records),
        "global_style_round_count": len(style_records),
        "combined_removed_round_count": len(combined_records),
        "tasks_with_removed_rounds": sum(1 for item in task_stats if item["removed_round_count"]),
        "tasks_with_no_remaining_rounds": sum(1 for item in task_stats if item["remaining_round_count"] == 0),
        "task_stats": task_stats,
    }
    (out / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "task_count": len(paths),
        "camera_motion_round_count": len(camera_records),
        "global_style_round_count": len(style_records),
        "combined_removed_round_count": len(combined_records),
        "tasks_with_no_remaining_rounds": manifest["tasks_with_no_remaining_rounds"],
        "output": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
