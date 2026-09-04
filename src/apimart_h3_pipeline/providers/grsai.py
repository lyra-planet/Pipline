"""GRSAI image editing client used to build static H3 references."""
from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .apimart import ApimartError, read_env_file, write_json

from ..core.policy import no_proxy_opener
from ..media import read_json


# GRSAI accepts a fixed set of aspect-ratio presets. Select the closest preset
# from the actual input image instead of asking the service to infer a ratio
# from the edited content.
SUPPORTED_ASPECT_RATIOS = (
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
)
GRSAI_CAPACITY_RETRY_SECONDS = 60
GRSAI_IMAGE_MODEL = "nano-banana-2"


def image_model_for_stage(stage_id: str) -> str:
    """Return the static-reference model assigned to a sequential stage.

    The stage argument is retained as part of the provider boundary so callers
    can select a model without changing their execution flow.  All image-edit
    stages intentionally use the same GRSAI model.
    """

    del stage_id
    return GRSAI_IMAGE_MODEL


def is_capacity_overload_error(error: BaseException) -> bool:
    """Return whether GRSAI reported a transient capacity rejection."""

    return "excessive system load" in str(error).casefold()


def aspect_ratio_for_image(image: Path) -> str:
    """Return the closest GRSAI-supported ratio for an input image."""

    try:
        with Image.open(image) as opened:
            width, height = opened.size
    except (OSError, ValueError) as error:
        raise ApimartError(f"cannot inspect image dimensions: {image}") from error
    if width <= 0 or height <= 0:
        raise ApimartError(f"image has invalid dimensions: {image}")
    ratio = width / height
    return min(
        SUPPORTED_ASPECT_RATIOS,
        key=lambda item: abs(math.log(ratio / item[1])),
    )[0]

def find_url(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("url", "image_url", "imageUrl"):
            item = value.get(key)
            if isinstance(item, str) and item.startswith(("https://", "http://", "data:")):
                return item
        for key in ("results", "data", "output"):
            found = find_url(value.get(key))
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = find_url(item)
            if found:
                return found
    return None


def find_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("id", "task_id", "taskId"):
            item = value.get(key)
            if isinstance(item, (str, int)) and str(item):
                return str(item)
        for key in ("data", "result", "output"):
            found = find_id(value.get(key))
            if found:
                return found
    return None


class GrsaiImageEditor:
    def __init__(self, env_file: Path, timeout_seconds: int = 1200) -> None:
        values = read_env_file(env_file)
        self.api_key = os.environ.get("GRSAI_API_KEY") or values.get("GRSAI_API_KEY", "")
        self.base_url = (os.environ.get("GRSAI_BASE_URL") or values.get("GRSAI_BASE_URL") or "https://grsaiapi.com").rstrip("/")
        if not self.api_key:
            raise ApimartError(f"GRSAI_API_KEY is absent from {env_file}")
        self.timeout_seconds = timeout_seconds
        self.model = GRSAI_IMAGE_MODEL
        self.opener = no_proxy_opener()

    def request(self, method: str, url: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=120) as response:
                value = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ApimartError(f"GRSAI HTTP {error.code}: {detail[:300]}") from error
        except urllib.error.URLError as error:
            raise ApimartError(f"GRSAI network error: {error.reason}") from error
        if not isinstance(value, dict):
            raise ApimartError("GRSAI response was not an object")
        return value

    @staticmethod
    def response_status(value: Mapping[str, Any]) -> str:
        """Return a normalized async-task status from common GRSAI response shapes."""
        for key in ("status", "state"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip().lower()
        for key in ("data", "result", "output"):
            item = value.get(key)
            if isinstance(item, Mapping):
                nested = GrsaiImageEditor.response_status(item)
                if nested:
                    return nested
        return ""

    @staticmethod
    def response_error(value: Mapping[str, Any]) -> str:
        for key in ("error", "message", "msg", "detail"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, Mapping):
                nested = GrsaiImageEditor.response_error(item)
                if nested:
                    return nested
        for key in ("data", "result", "output"):
            item = value.get(key)
            if isinstance(item, Mapping):
                nested = GrsaiImageEditor.response_error(item)
                if nested:
                    return nested
        return ""

    def edit(
        self,
        image: Path,
        raw_prompt: str,
        image_edit_prompt: str,
        output: Path,
        state_path: Path,
        style_reference: Path | None = None,
        aspect_ratio: str | None = None,
    ) -> dict[str, Any]:
        """Run an image edit, waiting through transient provider overloads."""

        while True:
            try:
                return self._edit_once(
                    image,
                    raw_prompt,
                    image_edit_prompt,
                    output,
                    state_path,
                    style_reference=style_reference,
                    aspect_ratio=aspect_ratio,
                )
            except ApimartError as error:
                if not is_capacity_overload_error(error):
                    raise
                waiting_state = read_json(state_path) if state_path.is_file() else {}
                waiting_state.update({
                    "status": "waiting_for_capacity",
                    "model": getattr(self, "model", GRSAI_IMAGE_MODEL),
                    "raw_prompt": raw_prompt,
                    "image_edit_prompt": image_edit_prompt,
                    "style_reference": str(style_reference) if style_reference else None,
                    "aspect_ratio": aspect_ratio or aspect_ratio_for_image(image),
                    "last_error": str(error),
                    "retry_after_seconds": GRSAI_CAPACITY_RETRY_SECONDS,
                })
                write_json(state_path, waiting_state)
                print(json.dumps({
                    "event": "grsai_capacity_wait",
                    "retry_after_seconds": GRSAI_CAPACITY_RETRY_SECONDS,
                    "error": str(error),
                }, ensure_ascii=False), flush=True)
                time.sleep(GRSAI_CAPACITY_RETRY_SECONDS)

    def _edit_once(
        self,
        image: Path,
        raw_prompt: str,
        image_edit_prompt: str,
        output: Path,
        state_path: Path,
        style_reference: Path | None = None,
        aspect_ratio: str | None = None,
    ) -> dict[str, Any]:
        aspect_ratio = aspect_ratio or aspect_ratio_for_image(image)
        model = getattr(self, "model", GRSAI_IMAGE_MODEL)
        persisted: dict[str, Any] = {}
        if state_path.is_file() and output.is_file() and output.stat().st_size:
            state = read_json(state_path)
            if (
                state.get("status") == "succeeded"
                and state.get("raw_prompt") == raw_prompt
                and state.get("image_edit_prompt") == image_edit_prompt
                and state.get("style_reference") == (str(style_reference) if style_reference else None)
                and state.get("aspect_ratio") == aspect_ratio
                and state.get("model", GRSAI_IMAGE_MODEL) == model
            ):
                return state
        if state_path.is_file():
            candidate = read_json(state_path)
            if (
                candidate.get("raw_prompt") == raw_prompt
                and candidate.get("image_edit_prompt") == image_edit_prompt
                and candidate.get("style_reference") == (str(style_reference) if style_reference else None)
                and candidate.get("aspect_ratio") == aspect_ratio
                and candidate.get("model", GRSAI_IMAGE_MODEL) == model
                and isinstance(candidate.get("task_id"), str)
                and candidate.get("status") in {"submitted", "queued", "processing", "running"}
            ):
                persisted = candidate
        if persisted:
            task_id = str(persisted["task_id"])
            url = find_url(persisted)
            initial_status = str(persisted.get("status", "submitted"))
            polls = int(persisted.get("polls", 0) or 0)
        else:
            image_data = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
            image_inputs = [image_data]
            if style_reference is not None:
                if not style_reference.is_file() or not style_reference.stat().st_size:
                    raise ApimartError(f"style master image is missing: {style_reference}")
                image_inputs.append("data:image/png;base64," + base64.b64encode(style_reference.read_bytes()).decode("ascii"))
            response = self.request(
                "POST", f"{self.base_url}/v1/api/generate",
                {"model": model, "prompt": image_edit_prompt, "images": image_inputs, "aspectRatio": aspect_ratio, "replyType": "async"},
            )
            initial_status = self.response_status(response)
            if initial_status in {"failed", "error", "cancelled", "canceled", "rejected", "expired"}:
                detail = self.response_error(response)
                failed_state = {
                    "status": "failed",
                    "model": model,
                    "raw_prompt": raw_prompt,
                    "image_edit_prompt": image_edit_prompt,
                    "style_reference": str(style_reference) if style_reference else None,
                    "aspect_ratio": aspect_ratio,
                    "error": detail or initial_status,
                }
                write_json(state_path, failed_state)
                raise ApimartError(
                    f"GRSAI image task failed immediately: status={initial_status}"
                    + (f" detail={detail[:300]}" if detail else "")
                )
            task_id = find_id(response)
            url = find_url(response)
            polls = 0
            if task_id:
                write_json(state_path, {
                    "status": initial_status or "submitted",
                    "model": model,
                    "raw_prompt": raw_prompt,
                    "image_edit_prompt": image_edit_prompt,
                    "style_reference": str(style_reference) if style_reference else None,
                    "aspect_ratio": aspect_ratio,
                    "task_id": task_id,
                    "polls": polls,
                })
        started = time.monotonic()
        while not url and task_id and time.monotonic() - started < self.timeout_seconds:
            time.sleep(5)
            polls += 1
            result = self.request("GET", f"{self.base_url}/v1/api/result?{urllib.parse.urlencode({'id': task_id})}")
            status = self.response_status(result)
            if status in {"failed", "error", "cancelled", "canceled", "rejected", "expired"}:
                detail = self.response_error(result)
                write_json(state_path, {
                    "status": "failed",
                    "model": model,
                    "raw_prompt": raw_prompt,
                    "image_edit_prompt": image_edit_prompt,
                    "style_reference": str(style_reference) if style_reference else None,
                    "task_id": task_id,
                    "polls": polls,
                    "aspect_ratio": aspect_ratio,
                    "error": detail or status,
                })
                raise ApimartError(
                    f"GRSAI image task failed: task_id={task_id} status={status}"
                    + (f" detail={detail[:300]}" if detail else "")
                )
            write_json(state_path, {
                "status": status or "processing",
                "model": model,
                "raw_prompt": raw_prompt,
                "image_edit_prompt": image_edit_prompt,
                "style_reference": str(style_reference) if style_reference else None,
                "aspect_ratio": aspect_ratio,
                "task_id": task_id,
                "polls": polls,
            })
            url = find_url(result)
        if not url:
            write_json(state_path, {
                "status": "timeout",
                "model": model,
                "raw_prompt": raw_prompt,
                "image_edit_prompt": image_edit_prompt,
                "task_id": task_id,
                "polls": polls,
                "aspect_ratio": aspect_ratio,
            })
            raise ApimartError(f"GRSAI image task produced no URL: task_id={task_id}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("data:"):
            output.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        else:
            with self.opener.open(urllib.request.Request(url), timeout=120) as response_stream:
                output.write_bytes(response_stream.read())
        with Image.open(output) as image_value:
            image_value.convert("RGB").save(output, "PNG")
        state = {
            "status": "succeeded",
            "model": model,
            "raw_prompt": raw_prompt,
            "image_edit_prompt": image_edit_prompt,
            "style_reference": str(style_reference) if style_reference else None,
            "aspect_ratio": aspect_ratio,
            "task_id": task_id,
            "polls": polls,
            "output": str(output),
        }
        write_json(state_path, state)
        return state
