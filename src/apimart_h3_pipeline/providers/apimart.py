#!/usr/bin/env python3
"""Small APIMart client for MiniMax-H3 generation and Context-IR.

The client reads credentials from an environment file or process environment,
never writes them to run artifacts. A changed request is submitted as a new
task; only an identical active task is resumed after an interruption.
Run it through ``with_apimart_proxy.sh`` on this host so standard HTTP proxy
variables are inherited by urllib.
"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ENV_FILE = Path(os.environ.get("APIMART_ENV_FILE", "~/.apimart.env")).expanduser()
DEFAULT_BASE_URL = "https://api.apimart.ai"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class ApimartError(RuntimeError):
    """A request or task failure with no credential-bearing context."""


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().removeprefix("export ")] = value.strip().strip("\"'")
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApimartError(f"state document is not an object: {path}")
    return value


def extract_error(body: bytes) -> str:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = body.decode("utf-8", errors="replace").strip()
        return text[:500] or "non-JSON error response"
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            message = error.get("message")
            kind = error.get("type")
            return f"{kind or 'api_error'} code={code!s} message={message!s}"
        for key in ("message", "detail", "msg", "error"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
        return json.dumps(parsed, ensure_ascii=False)[:500]
    return str(parsed)[:500]


def first_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value
    if isinstance(value, Mapping):
        for key in ("url", "video_url", "videoUrl", "download_url", "downloadUrl"):
            found = first_url(value.get(key))
            if found:
                return found
        for item in value.values():
            found = first_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = first_url(item)
            if found:
                return found
    return None


def task_id_from_submission(value: Mapping[str, Any], prefer_public_id: bool = False) -> str:
    data = value.get("data")
    if isinstance(data, list) and data and isinstance(data[0], Mapping):
        record = data[0]
        task_id = (record.get("id") or record.get("task_id")) if prefer_public_id else (record.get("task_id") or record.get("id"))
    elif isinstance(data, Mapping):
        task_id = (data.get("id") or data.get("task_id")) if prefer_public_id else (data.get("task_id") or data.get("id"))
    else:
        task_id = (value.get("id") or value.get("task_id")) if prefer_public_id else (value.get("task_id") or value.get("id"))
    if not isinstance(task_id, str) or not task_id:
        raise ApimartError("submission response did not include task_id")
    return task_id


def task_data(value: Mapping[str, Any]) -> Mapping[str, Any]:
    data = value.get("data", value)
    if not isinstance(data, Mapping):
        raise ApimartError("task status response did not contain an object")
    return data


class ApimartClient:
    def __init__(self, api_key: str, base_url: str, timeout_seconds: int) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # ProxyHandler() reads HTTP(S)_PROXY from the process environment.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler())
        self.direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def is_ctmoai(self) -> bool:
        """CTMOAI exposes the video route under singular ``/v1/video``."""

        return (urllib.parse.urlparse(self.base_url).hostname or "").lower() == "video.ctmoai.com"

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        # Only idempotent status requests are retried here. A failed POST can
        # have reached APIMart despite a broken response, so submission stays
        # single-shot and is protected by the persisted task state instead.
        retry_limit = 4 if method.upper() == "GET" else 1
        data = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        last_error: ApimartError | None = None
        for attempt in range(1, retry_limit + 1):
            request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
            except urllib.error.HTTPError as error:
                failure = ApimartError(f"HTTP {error.code}: {extract_error(error.read())}")
                if error.code < 500 or attempt == retry_limit:
                    raise failure from error
                last_error = failure
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                last_error = ApimartError(f"network request failed: {error}")
                if attempt == retry_limit:
                    raise last_error from error
            else:
                try:
                    value = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    last_error = ApimartError("API response was not valid JSON")
                    if attempt == retry_limit:
                        raise last_error from error
                else:
                    if not isinstance(value, dict):
                        raise ApimartError("API response was not an object")
                    return value
            time.sleep(attempt)
        raise ApimartError("API request retry loop exited unexpectedly") from last_error

    def upload_image(self, image: Path) -> dict[str, Any]:
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if image.stat().st_size > 20 * 1024 * 1024:
            raise ApimartError(f"image exceeds APIMart upload limit: {image}")
        content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        boundary = "----apimart" + secrets.token_hex(16)
        prefix = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{image.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        body = prefix + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        request = urllib.request.Request(
            self.base_url + "/v1/uploads/images", data=body, method="POST", headers=headers,
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ApimartError(f"image upload HTTP {error.code}: {extract_error(error.read())}") from error
        except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApimartError(f"image upload failed: {error}") from error
        if not isinstance(value, Mapping) or not isinstance(value.get("url"), str):
            raise ApimartError("image upload response did not contain url")
        return dict(value)

    def upload_media(self, media: Path, media_type: str) -> str:
        """Upload a local image/video to CTMOAI's stable media store."""

        if not self.is_ctmoai:
            raise ApimartError("stable media upload is only available on CTMOAI")
        if media_type not in {"images", "videos"}:
            raise ApimartError(f"unsupported CTMOAI media type: {media_type}")
        media = media.resolve()
        if not media.is_file() or not media.stat().st_size:
            raise FileNotFoundError(media)
        boundary = "----ctmoai" + secrets.token_hex(16)
        content_type = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\n{media_type}\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{media.name}\"\r\nContent-Type: {content_type}\r\n\r\n"
            ).encode(),
            media.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        request = urllib.request.Request(
            self.base_url + "/api/sd-media/upload",
            data=b"".join(parts),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with self.opener.open(request, timeout=max(self.timeout_seconds, 180)) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ApimartError(f"CTMOAI media upload HTTP {error.code}: {extract_error(error.read())}") from error
        except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApimartError(f"CTMOAI media upload failed: {error}") from error
        urls = value.get(media_type) if isinstance(value, Mapping) else None
        if not isinstance(urls, list) or not urls or not isinstance(urls[0], str) or not urls[0].strip():
            raise ApimartError(f"CTMOAI media upload response did not contain {media_type}[0]")
        url = urls[0].strip()
        # The upload edge and the video worker are separate services.  A URL
        # can be returned before the worker's object store has propagated it,
        # which otherwise appears as a misleading reference-video 404.  Probe
        # the returned object without creating another paid generation task.
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                probe = urllib.request.Request(url, headers={"Range": "bytes=0-1", "User-Agent": "ctmoai-media-probe/1"})
                with self.opener.open(probe, timeout=30) as response:
                    status = response.getcode()
                    if status in {200, 206}:
                        response.read(2)
                        # Allow a short edge-to-worker propagation window even
                        # when the public edge is already serving the object.
                        time.sleep(2)
                        return url
                    last_error = ApimartError(f"media probe returned HTTP {status}")
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                last_error = error
            if attempt < 6:
                time.sleep(attempt)
        raise ApimartError(f"uploaded CTMOAI media did not become readable: {url}") from last_error

    def submit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # CTMOAI's documented OpenAI Video-compatible route is plural
        # ``/v1/videos``. The older singular adapter accepted different field
        # names but did not reliably bind reference media to H3.
        path = "/v1/videos" if self.is_ctmoai else "/v1/videos/generations"
        return self.request_json("POST", path, payload)

    def task_status(self, task_id: str, language: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(task_id, safe="")
        if self.is_ctmoai:
            response = self.request_json("GET", f"/v1/videos/{encoded}")
            data = response.get("data") if isinstance(response, Mapping) else None
            record = data if isinstance(data, Mapping) and "status" in data else response
            raw_status = str(record.get("status", "")).strip().lower()
            status_map = {
                "success": "completed", "succeeded": "completed", "completed": "completed",
                "failure": "failed", "failed": "failed", "error": "failed",
                "cancelled": "cancelled", "canceled": "cancelled",
                "in_progress": "processing", "processing": "processing", "queued": "queued",
                "pending": "queued",
            }
            status = status_map.get(raw_status, raw_status)
            normalized: dict[str, Any] = {
                "status": status,
                "progress": record.get("progress"),
            }
            if status == "completed":
                video_url = first_url(record)
                if video_url:
                    normalized["result"] = {"videos": [{"url": video_url}]}
            elif status == "failed":
                normalized["error"] = record.get("fail_reason") or response.get("message")
            return normalized
        query = urllib.parse.urlencode({"language": language})
        return self.request_json("GET", f"/v1/tasks/{encoded}?{query}")

    def content_is_ready(self, task_id: str) -> bool:
        """Probe CTMOAI's content endpoint without downloading the video."""

        if not self.is_ctmoai:
            return False
        encoded = urllib.parse.quote(task_id, safe="")
        request = urllib.request.Request(
            f"{self.base_url}/v1/videos/{encoded}/content",
            headers={"Authorization": f"Bearer {self.api_key}", "Range": "bytes=0-0"},
        )
        try:
            with self.opener.open(request, timeout=min(max(self.timeout_seconds, 30), 60)) as response:
                if response.getcode() in {200, 206}:
                    response.read(1)
                    return True
                return False
        except urllib.error.HTTPError as error:
            # 400 is CTMOAI's documented response while the upstream task is
            # still queued; 404 can occur during object-store propagation.
            if error.code in {400, 404, 409}:
                return False
            raise ApimartError(f"CTMOAI content probe HTTP {error.code}: {extract_error(error.read())}") from error
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            return False

    def download(self, url: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial")
        hostname = (urllib.parse.urlparse(url).hostname or "").lower()
        # APIMart's control API needs Clash on this host, but the returned
        # getapib.org media CDN terminates the local proxy's TLS connection.
        # Keep task submission/polling proxied and fetch only this public CDN
        # directly; this does not alter an already submitted request.
        downloader = self.direct_opener if hostname == "getapib.org" or hostname.endswith(".getapib.org") else self.opener
        last_error: Exception | None = None
        expected_size: int | None = None
        for attempt in range(1, 9):
            offset = temporary.stat().st_size if temporary.is_file() else 0
            headers = {"User-Agent": "apimart-h3-client/1"}
            if hostname == "video.ctmoai.com":
                headers["Authorization"] = f"Bearer {self.api_key}"
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with downloader.open(request, timeout=max(self.timeout_seconds, 600)) as response:
                    status = response.getcode()
                    content_range = response.headers.get("Content-Range", "")
                    range_match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if status == 206:
                        if not range_match or int(range_match.group(1)) != offset:
                            raise ApimartError("video download returned an invalid byte range")
                        expected_size = int(range_match.group(3))
                        mode = "ab"
                    elif status == 200:
                        # A server that ignores Range requires a fresh local
                        # file so prior bytes are never duplicated.
                        offset = 0
                        expected_header = response.headers.get("Content-Length")
                        expected_size = int(expected_header) if expected_header and expected_header.isdigit() else None
                        mode = "wb"
                    else:
                        raise ApimartError(f"video download returned unexpected HTTP status {status}")
                    with temporary.open(mode) as handle:
                        shutil.copyfileobj(response, handle)
                received_size = temporary.stat().st_size
                if expected_size is not None and received_size != expected_size:
                    raise ApimartError(
                        f"video download is incomplete: received {received_size} of {expected_size} bytes"
                    )
                validation = video_summary(temporary)
                if validation.get("validation") != "ok":
                    temporary.unlink(missing_ok=True)
                    raise ApimartError("downloaded video failed ffprobe validation")
                temporary.replace(output)
                return
            except (ApimartError, urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                last_error = error
                received_size = temporary.stat().st_size if temporary.is_file() else 0
                print(json.dumps({
                    "event": "video_download_retry",
                    "attempt": attempt,
                    "received_bytes": received_size,
                    "expected_bytes": expected_size,
                    "error": str(error),
                }, ensure_ascii=False), flush=True)
                if attempt < 8:
                    time.sleep(attempt * 2)
        raise ApimartError("video download failed after 8 resumable attempts") from last_error


def media_payload(args: argparse.Namespace, ctmoai: bool = False) -> dict[str, Any]:
    first = args.first_frame_image
    last = args.last_frame_image
    image_urls = list(args.image_url or [])
    video_urls = list(args.video_url or [])
    audio_urls = list(args.audio_url or [])
    if first or last:
        if image_urls or video_urls or audio_urls:
            raise ApimartError("first/last-frame fields cannot be combined with reference media")
    if audio_urls and not (image_urls or video_urls):
        raise ApimartError("audio reference requires an image or video reference")
    if len(image_urls) > 9 or len(video_urls) > 3 or len(audio_urls) > 3:
        raise ApimartError("reference media exceeds APIMart item limits")
    payload: dict[str, Any] = {}
    if first and not ctmoai:
        payload["first_frame_image"] = first
    if last and not ctmoai:
        payload["last_frame_image"] = last
    if image_urls:
        payload["images" if ctmoai else "image_urls"] = image_urls
    if video_urls:
        payload["reference_videos" if ctmoai else "video_urls"] = video_urls
    if audio_urls:
        payload["reference_audios" if ctmoai else "audio_urls"] = audio_urls
    if ctmoai and (first or last):
        # CTMOAI uses the documented fl2v workflow for explicit first/last
        # frame inputs; a single first frame remains a one-element images list.
        payload["images"] = [item for item in (first, last) if item]
        payload["workflow_id"] = "fl2v"
    return payload


def build_payload(args: argparse.Namespace, model: str, ctmoai: bool = False) -> dict[str, Any]:
    prompt = (args.prompt or "").strip()
    if not prompt:
        raise ApimartError("prompt must be non-empty")
    if len(prompt) > 7000:
        raise ApimartError("prompt exceeds APIMart 7000-character limit")
    if not 4 <= args.duration <= 15:
        raise ApimartError("duration must be an integer from 4 to 15")
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": args.aspect_ratio,
    }
    # CTMOAI's OpenAI-video adapter calls the duration field ``seconds``;
    # APIMart's native route uses ``duration``.
    payload["seconds" if ctmoai else "duration"] = args.duration
    payload.update(media_payload(args, ctmoai=ctmoai))
    if ctmoai:
        # The public H3 768P contract for 16:9 is 1376x768. Keep the canvas
        # explicit instead of relying on a gateway default.
        if model.endswith("-768p") and args.aspect_ratio == "16:9":
            payload["size"] = "1376x768"
    if model == "MiniMax-H3":
        payload["resolution"] = args.resolution
        if args.watermark:
            payload["watermark"] = True
    return payload


def video_summary(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        return {"validation": "ffprobe_failed", "stderr": completed.stderr.strip()}
    try:
        return {"validation": "ok", "ffprobe": json.loads(completed.stdout)}
    except json.JSONDecodeError:
        return {"validation": "ffprobe_invalid_json"}


def wait_for_task(
    client: ApimartClient,
    state_path: Path,
    state: dict[str, Any],
    poll_seconds: float,
    deadline: float,
    language: str,
) -> dict[str, Any]:
    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ApimartError("state does not contain task_id")
    while True:
        try:
            response = client.task_status(task_id, language)
        except ApimartError as error:
            # A task ID is already durable. Keep polling it through transient
            # proxy/TLS failures rather than aborting a paid generation.
            if time.monotonic() >= deadline:
                raise ApimartError(f"task polling timed out after task_id={task_id}") from error
            print(json.dumps({
                "event": "task_poll_transport_retry",
                "task_id": task_id,
                "error": str(error),
            }, ensure_ascii=False), flush=True)
            time.sleep(min(max(poll_seconds, 1.0), 15.0))
            continue
        task = dict(task_data(response))
        state["last_status"] = task
        state["updated_at_unix"] = time.time()
        write_json(state_path, state)
        status = str(task.get("status", "")).lower()
        progress = task.get("progress")
        print(json.dumps({"event": "task_poll", "task_id": task_id, "status": status, "progress": progress}, ensure_ascii=False), flush=True)
        if status == "unknown" and client.content_is_ready(task_id):
            task["status"] = "completed"
            task["result"] = {"videos": [{"url": f"{client.base_url}/v1/videos/{urllib.parse.quote(task_id, safe='')}/content"}]}
            state["last_status"] = task
            write_json(state_path, state)
            print(json.dumps({"event": "task_content_ready", "task_id": task_id}, ensure_ascii=False), flush=True)
            return task
        if status in TERMINAL_STATUSES:
            return task
        if time.monotonic() >= deadline:
            raise ApimartError(f"task polling timed out after task_id={task_id}")
        time.sleep(poll_seconds)


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    values = read_env_file(args.env_file)
    api_key = (
        os.environ.get("APIMART_API_KEY")
        or values.get("APIMART_API_KEY", "")
        or os.environ.get("CTMOAI_API_KEY")
        or values.get("CTMOAI_API_KEY", "")
    )
    base_url = (
        args.base_url
        or os.environ.get("APIMART_BASE_URL")
        or values.get("APIMART_BASE_URL")
        or os.environ.get("CTMOAI_BASE_URL")
        or values.get("CTMOAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    if not api_key:
        raise ApimartError(f"APIMART_API_KEY is absent from environment and {args.env_file}")
    return api_key, base_url


def existing_or_new_state(state_path: Path, kind: str, payload: Mapping[str, Any], resubmit: bool) -> dict[str, Any]:
    """Reuse only the exact same active request; otherwise start a new task.

    The state file is only a network-recovery aid. It must never make a new
    prompt/media request poll an old paid task. ``resubmit`` remains in the
    signature for CLI compatibility, but changed/failed/incomplete requests
    are handled automatically.
    """
    if state_path.is_file():
        state = read_json(state_path)
        existing_payload = state.get("request") if state.get("kind") == kind else None
        if isinstance(existing_payload, Mapping):
            existing_fingerprint = json.dumps(
                existing_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            requested_fingerprint = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if existing_fingerprint == requested_fingerprint:
                task_id = state.get("task_id")
                if task_id:
                    last_status = state.get("last_status")
                    terminal_status = (
                        str(last_status.get("status", "")).lower()
                        if isinstance(last_status, Mapping)
                        else str(state.get("status", "")).lower()
                    )
                    if terminal_status not in {"failed", "cancelled"}:
                        return state
            # Different prompt/media/settings, a failed task, or an old state
            # without a request payload all start a fresh request.
    state = {
        "kind": kind,
        "submission_intent_at_unix": time.time(),
        "request": dict(payload),
        "task_id": None,
    }
    write_json(state_path, state)
    return state


def run_generation(args: argparse.Namespace, model: str, context_only: bool) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "apimart_task_state.json"
    api_key, base_url = resolve_credentials(args)
    client = ApimartClient(api_key, base_url, args.request_timeout)
    payload = build_payload(args, model, ctmoai=client.is_ctmoai)
    if args.dry_run:
        write_json(args.out_dir / "dry_run_payload.json", payload)
        print(json.dumps({"event": "dry_run", "payload_path": str(args.out_dir / 'dry_run_payload.json')}, ensure_ascii=False))
        return 0
    state = existing_or_new_state(state_path, "context_ir" if context_only else "generation", payload, args.resubmit)
    task_id = state.get("task_id")
    if not task_id:
        response = client.submit(payload)
        task_id = task_id_from_submission(response, prefer_public_id=client.is_ctmoai)
        state["task_id"] = task_id
        state["submitted_at_unix"] = time.time()
        state["submission"] = response
        write_json(state_path, state)
        print(json.dumps({"event": "task_submitted", "task_id": task_id, "model": model}, ensure_ascii=False), flush=True)
    task = wait_for_task(
        client, state_path, state, args.poll_seconds, time.monotonic() + args.total_timeout, args.language,
    )
    status = str(task.get("status", "")).lower()
    if status != "completed":
        raise ApimartError(f"task reached terminal status {status}: {task.get('error')}")
    result = task.get("result")
    if context_only:
        prompt = result.get("prompt") if isinstance(result, Mapping) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApimartError("Context-IR task completed without result.prompt")
        output = args.out_dir / "enhanced_prompt.txt"
        output.write_text(prompt.strip() + "\n", encoding="utf-8")
        state["output_prompt"] = str(output)
        write_json(state_path, state)
        print(json.dumps({"event": "context_completed", "task_id": task_id, "output": str(output)}, ensure_ascii=False))
        return 0
    output = args.out_dir / args.output_name
    if client.is_ctmoai:
        # CTMOAI documents the task content endpoint as the stable download
        # path; status responses may intentionally omit a CDN URL.
        client.download(
            f"{client.base_url}/v1/videos/{urllib.parse.quote(str(task_id), safe='')}/content",
            output,
        )
        video_url = f"{client.base_url}/v1/videos/{urllib.parse.quote(str(task_id), safe='')}/content"
    else:
        video_url = first_url(result.get("videos") if isinstance(result, Mapping) else None)
        if not video_url:
            raise ApimartError("H3 task completed without result.videos URL")
        client.download(video_url, output)
    state["output"] = str(output)
    state["video_url"] = video_url
    state["output_validation"] = video_summary(output)
    write_json(state_path, state)
    print(json.dumps({"event": "video_downloaded", "task_id": task_id, "output": str(output)}, ensure_ascii=False))
    return 0


def run_regeneration(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": "MiniMax-H3-Regeneration", "source_task_id": args.source_task_id}
    state_path = args.out_dir / "apimart_task_state.json"
    if args.dry_run:
        write_json(args.out_dir / "dry_run_payload.json", payload)
        return 0
    api_key, base_url = resolve_credentials(args)
    client = ApimartClient(api_key, base_url, args.request_timeout)
    state = existing_or_new_state(state_path, "regeneration", payload, args.resubmit)
    if not state.get("task_id"):
        response = client.submit(payload)
        state["task_id"] = task_id_from_submission(response, prefer_public_id=client.is_ctmoai)
        state["submitted_at_unix"] = time.time()
        state["submission"] = response
        write_json(state_path, state)
        print(json.dumps({"event": "task_submitted", "task_id": state["task_id"], "model": payload["model"]}, ensure_ascii=False), flush=True)
    task = wait_for_task(client, state_path, state, args.poll_seconds, time.monotonic() + args.total_timeout, args.language)
    if str(task.get("status", "")).lower() != "completed":
        raise ApimartError(f"task reached terminal status {task.get('status')}: {task.get('error')}")
    result = task.get("result")
    video_url = first_url(result.get("videos") if isinstance(result, Mapping) else None)
    if not video_url:
        raise ApimartError("regeneration completed without result.videos URL")
    output = args.out_dir / args.output_name
    client.download(video_url, output)
    state["output"] = str(output)
    state["video_url"] = video_url
    state["output_validation"] = video_summary(output)
    write_json(state_path, state)
    print(json.dumps({"event": "video_downloaded", "task_id": state["task_id"], "output": str(output)}, ensure_ascii=False))
    return 0


def run_upload_image(args: argparse.Namespace) -> int:
    api_key, base_url = resolve_credentials(args)
    client = ApimartClient(api_key, base_url, args.request_timeout)
    response = client.upload_image(args.image)
    record = {
        "local_image": str(args.image.resolve()),
        "url": response["url"],
        "filename": response.get("filename"),
        "bytes": response.get("bytes"),
        "created_at": response.get("created_at"),
    }
    if args.record:
        write_json(args.record, record)
    print(json.dumps(record, ensure_ascii=False))
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--base-url")
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--poll-seconds", type=float, default=7.0)
    parser.add_argument("--total-timeout", type=int, default=900)
    parser.add_argument("--language", choices=("zh", "en", "ko", "ja"), default="zh")


def add_media_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-url", action="append", default=[])
    parser.add_argument("--video-url", action="append", default=[])
    parser.add_argument("--audio-url", action="append", default=[])
    parser.add_argument("--first-frame-image")
    parser.add_argument("--last-frame-image")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="output.mp4")
    parser.add_argument("--resubmit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_media_args(parser)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="submit and wait for MiniMax-H3 generation")
    add_generation_args(generate)
    generate.add_argument("--model", default="MiniMax-H3")
    generate.add_argument("--resolution", choices=("768P", "2K"), default="768P")
    generate.add_argument("--watermark", action="store_true")

    context = subparsers.add_parser("context-ir", help="submit and wait for structured prompt enhancement")
    add_generation_args(context)

    regenerate = subparsers.add_parser("regenerate", help="regenerate own MiniMax-H3 768P task to 2K")
    regenerate.add_argument("--source-task-id", required=True)
    regenerate.add_argument("--out-dir", type=Path, required=True)
    regenerate.add_argument("--output-name", default="output_2k.mp4")
    regenerate.add_argument("--resubmit", action="store_true")
    regenerate.add_argument("--dry-run", action="store_true")

    upload = subparsers.add_parser("upload-image", help="upload image and emit reusable APIMart URL")
    upload.add_argument("--image", type=Path, required=True)
    upload.add_argument("--record", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        return run_generation(args, args.model, context_only=False)
    if args.command == "context-ir":
        return run_generation(args, "MiniMax-H3-Context-IR", context_only=True)
    if args.command == "regenerate":
        return run_regeneration(args)
    if args.command == "upload-image":
        return run_upload_image(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApimartError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"event": "fatal", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
