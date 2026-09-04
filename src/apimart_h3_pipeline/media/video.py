"""Video probing, H3 canvas normalization, and deterministic frame extraction."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from ..providers.apimart import ApimartError, write_json

from ..core.constants import H3_CANVAS_HEIGHT, H3_CANVAS_WIDTH, H3_FPS, H3_FRAME_COUNT

@dataclass(frozen=True)
class CanvasGeometry:
    """The reversible letterbox geometry used around every H3 request."""

    source_width: int
    source_height: int
    content_width: int
    content_height: int
    offset_x: int
    offset_y: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "mode": "letterbox_then_crop_v1",
            "source_width": self.source_width,
            "source_height": self.source_height,
            "canvas_width": H3_CANVAS_WIDTH,
            "canvas_height": H3_CANVAS_HEIGHT,
            "content_width": self.content_width,
            "content_height": self.content_height,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
        }


def compute_canvas_geometry(width: int, height: int) -> CanvasGeometry:
    """Fit a source rectangle into the H3 canvas without changing its aspect ratio."""

    if width <= 0 or height <= 0:
        raise ApimartError(f"source video has invalid dimensions: {width}x{height}")
    scale = min(H3_CANVAS_WIDTH / width, H3_CANVAS_HEIGHT / height)
    content_width = max(2, min(H3_CANVAS_WIDTH, int(width * scale) & ~1))
    content_height = max(2, min(H3_CANVAS_HEIGHT, int(height * scale) & ~1))
    return CanvasGeometry(
        source_width=width,
        source_height=height,
        content_width=content_width,
        content_height=content_height,
        offset_x=(H3_CANVAS_WIDTH - content_width) // 2,
        offset_y=(H3_CANVAS_HEIGHT - content_height) // 2,
    )
def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApimartError(f"expected JSON object: {path}")
    return value


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
            "-of", "json", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ApimartError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ApimartError(f"ffprobe emitted invalid JSON for {path}") from error
    if not isinstance(value, dict):
        raise ApimartError(f"ffprobe returned a non-object for {path}")
    return value


def stream_of(metadata: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    streams = metadata.get("streams")
    if not isinstance(streams, list):
        return None
    return next((item for item in streams if isinstance(item, Mapping) and item.get("codec_type") == kind), None)


def is_aligned_video(path: Path) -> bool:
    if not path.is_file() or not path.stat().st_size:
        return False
    try:
        video = stream_of(ffprobe(path), "video")
    except ApimartError:
        return False
    return bool(
        video
        and str(video.get("nb_frames", "")) == str(H3_FRAME_COUNT)
        and str(video.get("avg_frame_rate", "")) == f"{H3_FPS}/1"
    )


def has_audio(path: Path) -> bool:
    return stream_of(ffprobe(path), "audio") is not None


def is_h3_input_video(path: Path) -> bool:
    """The online H3 contract uses the same 1344x768 canvas as its 768P output."""

    if not path.is_file():
        return False
    try:
        video = stream_of(ffprobe(path), "video")
    except ApimartError:
        return False
    return bool(video and int(video.get("width", 0)) == 1344 and int(video.get("height", 0)) == 768)


def is_h3_generated_video(path: Path) -> bool:
    """Accept CTMOAI's 1376x768 16:9 raster as well as a native H3 canvas."""

    if not path.is_file():
        return False
    try:
        video = stream_of(ffprobe(path), "video")
    except ApimartError:
        return False
    return bool(
        video
        and int(video.get("height", 0)) == H3_CANVAS_HEIGHT
        and int(video.get("width", 0)) in {H3_CANVAS_WIDTH, 1376}
    )


def geometry_sidecar(path: Path) -> Path:
    return path.with_suffix(".geometry.json")


def source_canvas_geometry(source: Path) -> CanvasGeometry:
    video = stream_of(ffprobe(source), "video")
    if video is None:
        raise ApimartError(f"source video has no video stream: {source}")
    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ApimartError(f"source video has invalid dimensions: {source}") from error
    return compute_canvas_geometry(width, height)


def geometry_matches(path: Path, geometry: CanvasGeometry, role: str) -> bool:
    try:
        value = json.loads(geometry_sidecar(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("role") == role and value.get("geometry") == geometry.as_dict()


def write_geometry_sidecar(path: Path, geometry: CanvasGeometry, role: str) -> None:
    write_json(geometry_sidecar(path), {
        "kind": "h3_letterbox_geometry_v1",
        "role": role,
        "geometry": geometry.as_dict(),
    })


def load_geometry_sidecar(path: Path, role: str = "initial_input") -> CanvasGeometry:
    try:
        value = json.loads(geometry_sidecar(path).read_text(encoding="utf-8"))
        geometry = value["geometry"]
        if value.get("role") != role:
            raise ValueError(f"expected sidecar role {role!r}")
        return CanvasGeometry(
            source_width=int(geometry["source_width"]),
            source_height=int(geometry["source_height"]),
            content_width=int(geometry["content_width"]),
            content_height=int(geometry["content_height"]),
            offset_x=int(geometry["offset_x"]),
            offset_y=int(geometry["offset_y"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApimartError(f"invalid H3 geometry sidecar for {path}") from error


def run_ffmpeg(command: list[str], description: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise ApimartError(f"{description}: {completed.stderr.strip()}")


def materialize_initial_video(
    source: Path,
    target: Path,
    geometry: CanvasGeometry | None = None,
) -> Path:
    """Letterbox the source onto H3's canvas, preserving its aspect ratio."""

    geometry = geometry or source_canvas_geometry(source)
    if (
        is_aligned_video(target)
        and has_audio(target)
        and is_h3_input_video(target)
        and geometry_matches(target, geometry, "initial_input")
    ):
        return target
    metadata = ffprobe(source)
    try:
        duration = float(metadata["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ApimartError(f"source video has no positive duration: {source}") from error
    if duration <= 0:
        raise ApimartError(f"source video has no positive duration: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial.mp4")
    filter_graph = (
        f"scale={geometry.content_width}:{geometry.content_height}:flags=lanczos,"
        f"pad={H3_CANVAS_WIDTH}:{H3_CANVAS_HEIGHT}:{geometry.offset_x}:{geometry.offset_y}:color=black,"
        f"fps={H3_FRAME_COUNT}/{duration:.9f},trim=end_frame={H3_FRAME_COUNT},"
        f"tpad=stop_mode=clone:stop=2,setpts=N/{H3_FPS}/TB"
    )
    run_ffmpeg(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0", "-vf", filter_graph,
            "-frames:v", str(H3_FRAME_COUNT), "-r", str(H3_FPS),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(temporary),
        ],
        f"failed to normalize initial source {source}",
    )
    if not is_aligned_video(temporary) or not has_audio(temporary) or not is_h3_input_video(temporary):
        temporary.unlink(missing_ok=True)
        raise ApimartError(f"normalized source is not 1344x768/107-frame/24fps H.264+AAC: {source}")
    temporary.replace(target)
    write_geometry_sidecar(target, geometry, "initial_input")
    return target


def materialize_stage_video(
    source: Path,
    target: Path,
    geometry: CanvasGeometry | None = None,
) -> Path:
    """Publish a stage result without changing its pixels.

    Stage outputs are the immutable inputs to the next H3 request. Cropping,
    padding, or scaling here changes the visual state between edits, so all
    geometry restoration is deferred to :func:`materialize_final_video`.
    The provider must already have returned the native H3 canvas; incompatible
    output is rejected instead of being silently transformed.
    """

    geometry = geometry or source_canvas_geometry(source)
    if (
        target.is_file()
        and is_aligned_video(target)
        and is_h3_input_video(target)
        and geometry_matches(target, geometry, "stage_input")
    ):
        return target
    if not is_aligned_video(source) or not is_h3_generated_video(source):
        raise ApimartError(f"stage output is not a supported 768P raster with 107 frames at 24 fps: {source}")
    if not is_h3_input_video(source):
        raise ApimartError(
            f"stage output must already be 1344x768; refusing intermediate crop/scale: {source}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial.mp4")
    shutil.copy2(source, temporary)
    if not is_aligned_video(temporary) or not is_h3_input_video(temporary):
        temporary.unlink(missing_ok=True)
        raise ApimartError(f"published stage media is not 1344x768/107-frame/24fps: {source}")
    temporary.replace(target)
    write_geometry_sidecar(target, geometry, "stage_input")
    return target


def materialize_final_video(source: Path, target: Path, geometry: CanvasGeometry) -> Path:
    """Crop the H3 canvas and restore the source video's exact dimensions."""

    if not is_aligned_video(source) or not is_h3_input_video(source):
        raise ApimartError(f"final H3 media is not exactly 1344x768, 107 frames at 24 fps: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial.mp4")
    filter_graph = ",".join(
        (
            f"crop={geometry.content_width}:{geometry.content_height}:{geometry.offset_x}:{geometry.offset_y}",
            f"scale={geometry.source_width}:{geometry.source_height}:flags=lanczos",
        )
    )
    if has_audio(source):
        command = [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", filter_graph,
            "-frames:v", str(H3_FRAME_COUNT), "-r", str(H3_FPS),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(temporary),
        ]
    else:
        command = [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0", "-vf", filter_graph,
            "-frames:v", str(H3_FRAME_COUNT), "-r", str(H3_FPS),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(temporary),
        ]
    run_ffmpeg(command, f"failed to crop H3 canvas for {source}")
    if not is_aligned_video(temporary) or not has_audio(temporary):
        temporary.unlink(missing_ok=True)
        raise ApimartError(f"cropped final media is invalid: {source}")
    temporary.replace(target)
    write_geometry_sidecar(target, geometry, "final_output")
    return target


def select_keyframe(video: Path, output: Path, frame_index: int = 53) -> None:
    if not 0 <= frame_index < H3_FRAME_COUNT:
        raise ApimartError(f"reference frame index is outside the {H3_FRAME_COUNT}-frame contract: {frame_index}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.png")
    run_ffmpeg(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(temporary)],
        f"failed to extract frame {frame_index} from {video}",
    )
    with Image.open(temporary) as image:
        image.convert("RGB").save(output, "PNG")
    temporary.unlink(missing_ok=True)
