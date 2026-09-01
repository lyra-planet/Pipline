"""DashScope transport and multimodal input helpers."""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .apimart import ApimartError, read_env_file

from ..core.policy import no_proxy_opener, normalized_prompt
from ..core.constants import QWEN_CONTEXT_FRAME_INDICES
from ..resources.catalog import render_prompt


class DashScopeClient:
    """Provider transport shared by reference planning and observation."""

    def __init__(self, env_file: Path, base_url: str, model: str, timeout_seconds: int) -> None:
        values = read_env_file(env_file)
        self.api_key = os.environ.get("DASHSCOPE_API_KEY") or values.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ApimartError(f"DASHSCOPE_API_KEY is absent from {env_file}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model.strip()
        if not self.model or self.model.lower() in {"qwen-max", "qwen3-max"}:
            raise ApimartError("Qwen Max-tier models are not allowed for transition refinement")
        self.timeout_seconds = timeout_seconds
        self.opener = no_proxy_opener()

    @staticmethod
    def image_data_url(keyframe: Path) -> str:
        with Image.open(keyframe) as image:
            converted = image.convert("RGB")
            converted.thumbnail((768, 768))
            encoded = io.BytesIO()
            converted.save(encoded, "JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

    def complete(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            request = urllib.request.Request(
                self.url, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    result = json.load(response)
                if not isinstance(result, Mapping):
                    raise ApimartError("DashScope returned a non-object response")
                choices = result.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                    raise ApimartError("DashScope returned no completion choices")
                message = choices[0].get("message")
                content = message.get("content") if isinstance(message, Mapping) else None
                if not isinstance(content, str):
                    raise ApimartError("DashScope returned no message content")
                return content, result
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code < 500 or attempt == 3:
                    raise ApimartError(f"DashScope HTTP {error.code}: {detail[:300]}") from error
                last_error = error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(attempt)
        raise ApimartError("DashScope transition refinement failed after retries") from last_error

    @staticmethod
    def validate_context_frames(frames: Sequence[Path]) -> None:
        if len(frames) != len(QWEN_CONTEXT_FRAME_INDICES):
            raise ApimartError(
                "Qwen-VL prompt refinement requires five parent-video frames: "
                f"{len(frames)} != {len(QWEN_CONTEXT_FRAME_INDICES)}"
            )
        if not all(frame.is_file() and frame.stat().st_size for frame in frames):
            raise ApimartError("Qwen-VL prompt refinement received a missing parent-video frame")

    def multimodal_content(
        self,
        text: str,
        context_frames: Sequence[Path],
        style_references: Sequence[Path] = (),
        reference_roles: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        self.validate_context_frames(context_frames)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame_index, frame in zip(QWEN_CONTEXT_FRAME_INDICES, context_frames, strict=True):
            content.append({
                "type": "text",
                "text": render_prompt("qwen_parent_frame_label.txt", frame_index=frame_index),
            })
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(frame)}})
        if reference_roles and len(reference_roles) != len(style_references):
            raise ApimartError(
                "Qwen-VL reference role count does not match attached pictures: "
                f"{len(reference_roles)} != {len(style_references)}"
            )
        for picture_index, reference in enumerate(style_references, 1):
            if not reference.is_file() or not reference.stat().st_size:
                raise ApimartError(f"Qwen-VL style reference is missing: {reference}")
            role = reference_roles[picture_index - 1] if reference_roles else {}
            if role:
                role_name = normalized_prompt(str(role.get("role", "")))
                source_frame = role.get("source_frame_index")
                if not role_name or not isinstance(source_frame, int):
                    raise ApimartError(f"invalid Qwen-VL reference role for Picture {picture_index}")
                label = render_prompt(
                    "qwen_picture_role_label.txt",
                    picture_index=picture_index,
                    role_name=role_name,
                    source_frame=source_frame,
                )
            else:
                label = render_prompt("qwen_picture_label.txt", picture_index=picture_index)
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": self.image_data_url(reference)}})
        return content
