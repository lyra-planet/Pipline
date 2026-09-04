"""Geometry-preserving image preparation for H3 reference editing."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from ..core.constants import H3_CANVAS_HEIGHT, H3_CANVAS_WIDTH
from ..providers.apimart import ApimartError
from .video import CanvasGeometry


def prepare_image_edit_input(
    canvas_image: Path,
    output: Path,
    geometry: CanvasGeometry,
) -> Path:
    """Remove H3 padding and restore the source raster before image editing."""

    with Image.open(canvas_image) as opened:
        image = opened.convert("RGB")
    if image.size != (H3_CANVAS_WIDTH, H3_CANVAS_HEIGHT):
        raise ApimartError(
            f"reference source is not the H3 canvas {H3_CANVAS_WIDTH}x{H3_CANVAS_HEIGHT}: "
            f"{canvas_image} is {image.width}x{image.height}"
        )
    content = image.crop((
        geometry.offset_x,
        geometry.offset_y,
        geometry.offset_x + geometry.content_width,
        geometry.offset_y + geometry.content_height,
    ))
    if content.size != (geometry.source_width, geometry.source_height):
        content = content.resize(
            (geometry.source_width, geometry.source_height),
            Image.Resampling.LANCZOS,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    content.save(output, "PNG")
    return output


def materialize_h3_reference_image(
    edited_image: Path,
    output: Path,
    geometry: CanvasGeometry,
) -> Path:
    """Fit an edited source image into its exact H3 content rectangle."""

    with Image.open(edited_image) as opened:
        image = opened.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ApimartError(f"edited reference image has invalid dimensions: {edited_image}")
    source_aligned = ImageOps.fit(
        image,
        (geometry.source_width, geometry.source_height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    content = source_aligned.resize(
        (geometry.content_width, geometry.content_height),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (H3_CANVAS_WIDTH, H3_CANVAS_HEIGHT), "black")
    canvas.paste(content, (geometry.offset_x, geometry.offset_y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "PNG")
    return output
