#!/usr/bin/env python3
"""Audit compact plans and emit conservative Qwen repair suggestions."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM = r"""You audit video-editing diffusion plans. Compare the original instruction
with every ordered video_diffusion_rounds instruction and report ONLY definite
semantic errors, not wording preferences. A valid plan may combine clauses in
one round and may use synonyms such as move/pan, change/transform, add/introduce,
or swap/replace when the requested result is unchanged. Do not require the
global phrase 'maintain all other elements unchanged' to be repeated in every
round. Do not reorder independent operations merely to match prose order.

Flag an issue only when the plan definitely changes the requested operation or
loses a concrete requirement, for example:
- an explicit Replace of an existing visible object/background/scene/environment
  is rewritten as Transform or Change, so a hard replacement can become a
  modification (this direction is important; do not flag the reverse by itself);
- a concrete requested operation, target, or negative constraint is omitted;
- an unrequested operation or target is added;
- round order makes an explicit dependency impossible.
Do NOT flag camera, zoom, pan, style, lighting, season, or action operations just
because they use transform/change. Do not flag a specific action described by an
equivalent result phrase (for example 'wall into rubble') unless the requested
motion/event is definitely absent. Preserve all valid constraints and make only
the smallest correction. Do not invent visual facts. If evidence is ambiguous,
mark has_problem false and leave the rounds unchanged.

Return JSON only with exactly these keys:
{"has_problem": boolean, "issues": [{"type": string, "round": integer|null,
"evidence": string, "severity": "high"|"medium"|"low"}],
"repaired_rounds": [{"round": integer, "instruction": string}],
"repair_summary": string}
repaired_rounds must have exactly the same round count, numbers, and order as
the input. If no problem exists, copy every input instruction exactly, use an
empty issues list, and use an empty repair_summary."""


def load_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DASHSCOPE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("DASHSCOPE_API_KEY", "")


def parse_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def ask(key: str, model: str, original: str, rounds: list[dict[str, Any]],
        timeout: int, retries: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "original_complex_instruction": original,
                "video_diffusion_rounds": rounds,
            }, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=data, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            return parse_object(result["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Qwen call failed: {last}")


def validate(input_rounds: list[dict[str, Any]], result: dict[str, Any]) -> str | None:
    repaired = result.get("repaired_rounds")
    if not isinstance(repaired, list) or len(repaired) != len(input_rounds):
        return "repaired_rounds length differs from input"
    expected = [int(item.get("round")) for item in input_rounds]
    actual: list[int] = []
    for item in repaired:
        if not isinstance(item, dict) or not isinstance(item.get("instruction"), str):
            return "invalid repaired round"
        try:
            actual.append(int(item.get("round")))
        except (TypeError, ValueError):
            return "invalid repaired round number"
    if actual != expected:
        return "round numbers/order changed"
    if not isinstance(result.get("has_problem"), bool) or not isinstance(result.get("issues"), list):
        return "invalid result schema"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("/root/.dashscope.env"))
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    key = load_key(args.env_file)
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is missing")
    paths = sorted(args.compact_dir.glob("task_*.json"))
    records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    output: dict[str, Any] = {"model": args.model, "source_dir": str(args.compact_dir),
                              "tasks": [], "errors": []}

    def process(item: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
        path, record = item
        rounds = record.get("video_diffusion_rounds")
        if not isinstance(rounds, list) or not rounds:
            return {"task_id": str(record.get("task_id", "")), "file": path.name,
                    "error": "missing video_diffusion_rounds"}
        try:
            result = ask(key, args.model, str(record.get("original_complex_instruction", "")),
                         rounds, args.timeout, args.retries)
            error = validate(rounds, result)
            base = {"task_id": str(record.get("task_id", "")), "file": path.name,
                    "original_complex_instruction": record.get("original_complex_instruction", ""),
                    "original_rounds": rounds}
            if error:
                base.update({"error": error, "qwen": result})
            else:
                base.update(result)
            return base
        except Exception as exc:
            return {"task_id": str(record.get("task_id", "")), "file": path.name,
                    "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, item) for item in records]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item = future.result()
            (output["errors"] if "error" in item else output["tasks"]).append(item)
            if index % 25 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}", flush=True)
    output["tasks"].sort(key=lambda x: int(x["task_id"]))
    output["errors"].sort(key=lambda x: int(x.get("task_id") or 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(output["tasks"]), "errors": len(output["errors"]),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if not output["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
