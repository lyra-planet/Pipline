from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from apimart_h3_pipeline.media import video as media_video
from apimart_h3_pipeline.media.images import materialize_h3_reference_image, prepare_image_edit_input


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_apimart_minimax_h3_sequential.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_apimart_minimax_h3_sequential_geometry", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class H3CanvasGeometryTests(unittest.TestCase):
    def test_near_square_source_is_letterboxed_without_stretching(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        self.assertEqual(geometry.content_width, 766)
        self.assertEqual(geometry.content_height, 768)
        self.assertEqual(geometry.offset_x, 289)
        self.assertEqual(geometry.offset_y, 0)
        self.assertEqual(geometry.as_dict()["mode"], "letterbox_then_crop_v1")

    def test_wide_source_gets_top_and_bottom_padding(self) -> None:
        geometry = module.compute_canvas_geometry(1920, 1080)
        self.assertEqual((geometry.content_width, geometry.content_height), (1344, 756))
        self.assertEqual((geometry.offset_x, geometry.offset_y), (0, 6))

    def test_native_h3_geometry_has_no_padding(self) -> None:
        geometry = module.compute_canvas_geometry(1344, 768)
        self.assertEqual((geometry.content_width, geometry.content_height), (1344, 768))
        self.assertEqual((geometry.offset_x, geometry.offset_y), (0, 0))

    def test_grsai_aspect_ratio_follows_input_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wide = root / "wide.png"
            square = root / "square.png"
            Image.new("RGB", (1344, 768), "black").save(wide)
            Image.new("RGB", (1016, 1018), "black").save(square)
            self.assertEqual(module.aspect_ratio_for_image(wide), "16:9")
            self.assertEqual(module.aspect_ratio_for_image(square), "1:1")

    def test_final_output_crops_and_scales_to_original_dimensions(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "h3_output.mp4"
            target = root / "final.mp4"
            source.write_bytes(b"source")

            def fake_ffmpeg(command, _description):
                Path(command[-1]).write_bytes(b"encoded")

            with (
                patch.object(media_video, "is_aligned_video", return_value=True),
                patch.object(media_video, "is_h3_input_video", return_value=True),
                patch.object(media_video, "has_audio", side_effect=[False, True]),
                patch.object(media_video, "run_ffmpeg", side_effect=fake_ffmpeg) as ffmpeg,
                patch.object(media_video, "write_geometry_sidecar"),
            ):
                result = module.materialize_final_video(source, target, geometry)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"encoded")
            command = ffmpeg.call_args.args[0]
            filter_index = command.index("-vf") + 1
            self.assertEqual(
                command[filter_index],
                "crop=766:768:289:0,scale=1016:1018:flags=lanczos",
            )

    def test_stage_publish_preserves_provider_output_without_geometry_transform(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "h3_output.mp4"
            target = root / "published.mp4"
            source.write_bytes(b"provider-output-without-reencoding")

            with (
                patch.object(media_video, "is_aligned_video", return_value=True),
                patch.object(media_video, "is_h3_generated_video", return_value=True),
                patch.object(media_video, "is_h3_input_video", return_value=True),
                patch.object(media_video, "has_audio", return_value=False),
                patch.object(media_video, "geometry_matches", return_value=False),
                patch.object(media_video, "write_geometry_sidecar") as write_sidecar,
                patch.object(media_video, "run_ffmpeg") as ffmpeg,
            ):
                result = module.materialize_stage_video(source, target, geometry)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            ffmpeg.assert_not_called()
            write_sidecar.assert_called_once_with(target, geometry, "stage_input")

    def test_stage_publish_rejects_non_native_provider_output(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wide_h3_output.mp4"
            target = root / "published.mp4"
            source.write_bytes(b"provider-output")

            with (
                patch.object(media_video, "is_aligned_video", return_value=True),
                patch.object(media_video, "is_h3_generated_video", return_value=True),
                patch.object(media_video, "is_h3_input_video", return_value=False),
            ):
                with self.assertRaisesRegex(module.ApimartError, "refusing intermediate crop/scale"):
                    module.materialize_stage_video(source, target, geometry)
            self.assertFalse(target.exists())

    def test_reference_edit_round_trip_removes_and_restores_letterbox(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canvas = root / "canvas.png"
            edit_input = root / "edit_input.png"
            edited = root / "edited.png"
            reference = root / "reference.png"

            image = Image.new("RGB", (1344, 768), "black")
            image.paste((220, 80, 40), (geometry.offset_x, 0, geometry.offset_x + geometry.content_width, geometry.content_height))
            image.save(canvas)
            prepare_image_edit_input(canvas, edit_input, geometry)
            with Image.open(edit_input) as prepared:
                self.assertEqual(prepared.size, (1016, 1018))
                self.assertEqual(prepared.getpixel((0, 0)), (220, 80, 40))

            Image.new("RGB", (1024, 1024), (30, 180, 90)).save(edited)
            materialize_h3_reference_image(edited, reference, geometry)
            with Image.open(reference) as restored:
                self.assertEqual(restored.size, (1344, 768))
                self.assertEqual(restored.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(restored.getpixel((geometry.offset_x, 0)), (30, 180, 90))
                self.assertEqual(restored.getpixel((geometry.offset_x + geometry.content_width - 1, 767)), (30, 180, 90))


if __name__ == "__main__":
    unittest.main()
