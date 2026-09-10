#!/usr/bin/env python3
"""Apply reviewed, non-reordering compact-plan repairs."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


# These are explicit Replace operations whose first plan round weakened the
# verb to Transform. Camera/style/action Transform rounds are intentionally not
# included.
BACKGROUND_REPLACE_IDS = {
    1, 2, 5, 6, 7, 8,
    259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 270, 271,
    272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
    285, 287, 288, 457, 470,
}


def replace_transform(text: str) -> str:
    match = re.match(r"^(Transform\s+.+?)\s+(into|to)\s+(.+)$", text.strip(), re.S)
    if not match:
        raise ValueError(f"cannot convert background Transform wording: {text}")
    return f"Replace {match.group(1)[len('Transform '):]} with {match.group(3)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.backup.exists():
        raise SystemExit(f"backup already exists: {args.backup}")
    shutil.copytree(args.source, args.backup)
    changes: list[dict[str, object]] = []
    for path in sorted(args.source.glob("task_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        task_id = int(data["task_id"])
        rounds = data.get("video_diffusion_rounds")
        if not isinstance(rounds, list):
            raise SystemExit(f"{path.name}: rounds is not a list")
        original = json.loads(json.dumps(rounds, ensure_ascii=False))
        if task_id in BACKGROUND_REPLACE_IDS:
            rounds[0]["instruction"] = replace_transform(str(rounds[0]["instruction"]))
        special: dict[int, dict[int, str]] = {
            7: {1: "Replace the background scene with a dry, rocky desert landscape featuring scattered rocks and sparse dry shrubs. Keep the iguana and the stone under the iguana unchanged."},
            225: {1: "Remove the on-screen text 'A 1' from the center of the frame with no ghosting, partial letters, or flicker; keep the bottom caption line and all orchard, subject, camera, and background motion intact. Maintain all other elements unchanged."},
            263: {3: "Change the woman's lipstick color to deep plum while keeping her hand gestures and pan movement. Keep all other elements unchanged."},
            345: {2: "Change the woman's action from placing her hands on the railing to turning 180 degrees and walking away from the camera toward the background."},
            539: {1: "Keep the tourists, stone observation deck, low wall, and the original slow zoom-in camera motion unchanged."},
            595: {2: "Transform the video into a Claymation-style animation. Render all people, the ceremony structure/altar, chairs, and the surrounding nature as textured plasticine clay figures/props with visible hand-molded qualities."},
            599: {4: "Replace the human arm (with wristwatch) with an astronaut's white space suit arm and glove."},
        }
        for round_number, instruction in special.get(task_id, {}).items():
            rounds[round_number - 1]["instruction"] = instruction
        if rounds != original:
            data["video_diffusion_rounds"] = rounds
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changes.append({"task_id": str(task_id), "file": path.name,
                            "changed_rounds": [i + 1 for i, (a, b) in enumerate(zip(original, rounds)) if a != b]})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "source": str(args.source), "backup": str(args.backup),
        "no_reorder": True, "changed_tasks": changes,
        "changed_task_count": len(changes),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_tasks": len(changes), "manifest": str(args.manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
