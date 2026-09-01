"""Local MiniMax-H3 execution through the standard ComfyUI HTTP API.

The local backend consumes a user-supplied API-format workflow template.  It
discovers the H3, LoadVideo, LoadImage, and SaveVideo nodes by ``class_type``
instead of relying on machine-specific node numbers, then submits the same
prompt and reference images produced by the shared bridge.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.constants import H3_CANVAS_HEIGHT, H3_CANVAS_WIDTH, H3_FRAME_COUNT
from ..providers.apimart import ApimartError, write_json


class LocalH3Error(ApimartError):
    """Raised when the local ComfyUI H3 backend cannot complete a request."""


@dataclass(frozen=True)
class LocalH3Config:
    """Connection and filesystem contract for one local ComfyUI instance."""

    server: str
    workflow_template: Path
    input_dir: Path
    output_dir: Path
    timeout_seconds: int
    poll_seconds: float

    def __post_init__(self) -> None:
        if not self.server.strip():
            raise LocalH3Error("local H3 server URL is required")
        if self.timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise LocalH3Error("local H3 timeout values must be positive")


class LocalH3MediaAdapter:
    """Bridge-compatible media adapter that exposes local image file URLs."""

    is_ctmoai = False
    base_url = "local://"

    @staticmethod
    def upload_image(image: Path) -> dict[str, str]:
        resolved = image.resolve()
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise LocalH3Error(f"local reference image is missing: {resolved}")
        return {"url": resolved.as_uri(), "upload_mode": "local_file"}


class LocalH3Client:
    """Submit one H3 graph to a local ComfyUI server and collect its video."""

    def __init__(self, config: LocalH3Config) -> None:
        self.config = config
        self.server = config.server.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _node_class(node: Mapping[str, Any]) -> str:
        value = node.get("class_type")
        return value if isinstance(value, str) else ""

    @classmethod
    def _find_node(
        cls,
        graph: Mapping[str, Mapping[str, Any]],
        predicate,
        label: str,
    ) -> tuple[str, dict[str, Any]]:
        matches = [
            (str(node_id), dict(node))
            for node_id, node in graph.items()
            if isinstance(node, Mapping) and predicate(cls._node_class(node))
        ]
        if len(matches) != 1:
            raise LocalH3Error(f"workflow must contain exactly one {label} node; found {len(matches)}")
        return matches[0]

    @staticmethod
    def _new_node_id(graph: Mapping[str, Any]) -> str:
        numeric = [int(str(node_id)) for node_id in graph if str(node_id).isdigit()]
        candidate = max(numeric, default=0) + 1
        while str(candidate) in graph:
            candidate += 1
        return str(candidate)

    def load_template(self) -> dict[str, dict[str, Any]]:
        path = self.config.workflow_template.resolve()
        if not path.is_file():
            raise LocalH3Error(f"local H3 workflow template is missing: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LocalH3Error(f"could not read local H3 workflow template: {path}") from error
        if not isinstance(value, dict) or not value:
            raise LocalH3Error("local H3 workflow template must be a non-empty API-format graph")
        if not all(isinstance(node, Mapping) for node in value.values()):
            raise LocalH3Error("local H3 workflow template nodes must be objects")
        return {str(node_id): dict(node) for node_id, node in value.items()}

    def build_workflow(
        self,
        prompt: str,
        input_name: str,
        reference_names: Sequence[str],
        output_prefix: str,
    ) -> dict[str, dict[str, Any]]:
        graph = copy.deepcopy(self.load_template())
        h3_id, h3_node = self._find_node(
            graph,
            lambda class_type: "ReferenceToVideo" in class_type,
            "MiniMax-H3 reference-to-video",
        )
        load_video_id, load_video_node = self._find_node(
            graph,
            lambda class_type: class_type == "LoadVideo",
            "LoadVideo",
        )
        save_video_id, save_video_node = self._find_node(
            graph,
            lambda class_type: class_type == "SaveVideo",
            "SaveVideo",
        )
        h3_inputs = h3_node.setdefault("inputs", {})
        load_video_node.setdefault("inputs", {})["file"] = input_name
        h3_inputs.update({
            "prompt": prompt,
            "width": H3_CANVAS_WIDTH,
            "height": H3_CANVAS_HEIGHT,
            "length": H3_FRAME_COUNT,
        })
        if "ref_videos.ref_video_0" not in h3_inputs:
            h3_inputs["ref_videos.ref_video_0"] = [load_video_id, 0]

        image_nodes = [
            str(node_id)
            for node_id, node in graph.items()
            if isinstance(node, Mapping) and self._node_class(node) == "LoadImage"
        ]
        for key in tuple(h3_inputs):
            if str(key).startswith("ref_images."):
                h3_inputs.pop(key)
        if not reference_names:
            for node_id in image_nodes:
                graph.pop(node_id, None)
        else:
            if not image_nodes:
                raise LocalH3Error("workflow template lacks a LoadImage node for reference images")
            template_image_id = image_nodes[0]
            for index, image_name in enumerate(reference_names):
                node_id = template_image_id if index == 0 else self._new_node_id(graph)
                if index:
                    graph[node_id] = copy.deepcopy(graph[template_image_id])
                graph[node_id].setdefault("inputs", {})["image"] = image_name
                h3_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        save_video_node.setdefault("inputs", {})["filename_prefix"] = output_prefix
        graph[h3_id] = h3_node
        graph[load_video_id] = load_video_node
        graph[save_video_id] = save_video_node
        return graph

    def _request_json(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = urllib.request.Request(self.server + path, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with self._opener.open(request, timeout=min(60, self.config.timeout_seconds)) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise LocalH3Error(f"local ComfyUI request failed at {path}: {error}") from error
        if not isinstance(value, dict):
            raise LocalH3Error(f"local ComfyUI response at {path} is not an object")
        return value

    def _materialize_input(self, source: Path, token: str) -> str:
        source = source.resolve()
        if not source.is_file():
            raise LocalH3Error(f"local H3 source video is missing: {source}")
        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        target = self.config.input_dir / f"{token}_input.mp4"
        if source != target:
            shutil.copy2(source, target)
        return target.name

    def _materialize_references(self, references: Sequence[Path], token: str) -> list[str]:
        self.config.input_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for index, reference in enumerate(references, 1):
            source = reference.resolve()
            if not source.is_file():
                raise LocalH3Error(f"local H3 reference image is missing: {source}")
            target = self.config.input_dir / f"{token}_reference_{index}{source.suffix or '.png'}"
            if source != target:
                shutil.copy2(source, target)
            names.append(target.name)
        return names

    @staticmethod
    def _history_output(history: Mapping[str, Any], output_dir: Path) -> Path:
        outputs = history.get("outputs")
        if not isinstance(outputs, Mapping):
            raise LocalH3Error("local ComfyUI history has no outputs")
        for node_output in outputs.values():
            if not isinstance(node_output, Mapping):
                continue
            for key in ("videos", "gifs", "images", "audio"):
                items = node_output.get(key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    filename = item.get("filename")
                    subfolder = item.get("subfolder", "")
                    if not isinstance(filename, str) or not isinstance(subfolder, str):
                        continue
                    candidate = (output_dir / subfolder / filename).resolve()
                    if output_dir.resolve() in candidate.parents and candidate.is_file() and candidate.stat().st_size:
                        return candidate
        raise LocalH3Error("local ComfyUI history has no readable video output")

    def _wait_for_output(self, prompt_id: str) -> Path:
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            history = self._request_json(f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            if isinstance(entry, Mapping):
                status = entry.get("status")
                if isinstance(status, Mapping) and status.get("status_str") in {"error", "failed"}:
                    raise LocalH3Error(f"local ComfyUI job {prompt_id} failed: {status.get('status_str')}")
                if isinstance(status, Mapping) and status.get("completed"):
                    if status.get("status_str") != "success":
                        raise LocalH3Error(f"local ComfyUI job {prompt_id} ended with {status.get('status_str')}")
                    return self._history_output(entry, self.config.output_dir)
            time.sleep(self.config.poll_seconds)
        raise LocalH3Error(f"local ComfyUI job timed out after {self.config.timeout_seconds} seconds: {prompt_id}")

    def generate(
        self,
        *,
        source_video: Path,
        prompt: str,
        reference_images: Sequence[Path],
        destination: Path,
        stage_dir: Path,
        stage_id: str,
    ) -> Path:
        token = stage_id or "stage"
        input_name = self._materialize_input(source_video, token)
        reference_names = self._materialize_references(reference_images, token)
        graph = self.build_workflow(prompt, input_name, reference_names, token)
        stage_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = stage_dir / "local_workflow.json"
        write_json(workflow_path, graph)
        state_path = stage_dir / "local_task_state.json"
        request = {"prompt": prompt, "input_name": input_name, "reference_names": reference_names}
        state: dict[str, Any] = {}
        if state_path.is_file():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict) and loaded.get("request") == request and isinstance(loaded.get("prompt_id"), str):
                state = loaded
        prompt_id = state.get("prompt_id")
        if not isinstance(prompt_id, str):
            submitted = self._request_json("/prompt", {"prompt": graph})
            prompt_id = submitted.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise LocalH3Error(f"local ComfyUI rejected workflow: {submitted}")
            state = {"request": request, "prompt_id": prompt_id, "workflow": str(workflow_path)}
            write_json(state_path, state)
        generated = self._wait_for_output(prompt_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if generated.resolve() != destination.resolve():
            shutil.copy2(generated, destination)
        state["output"] = str(generated)
        write_json(state_path, state)
        return destination


__all__ = ["LocalH3Client", "LocalH3Config", "LocalH3Error", "LocalH3MediaAdapter"]
