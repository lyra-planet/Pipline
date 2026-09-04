from __future__ import annotations

import json
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from apimart_h3_pipeline.providers import grsai
from apimart_h3_pipeline.providers.apimart import ApimartError


class GrsaiCapacityRetryTests(unittest.TestCase):
    def test_capacity_failure_waits_one_minute_then_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            state_path = root / "state.json"
            image.write_bytes(b"input")
            state_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            editor = grsai.GrsaiImageEditor.__new__(grsai.GrsaiImageEditor)
            succeeded = {"status": "succeeded", "output": str(root / "output.png")}
            with (
                patch.object(
                    editor,
                    "_edit_once",
                    side_effect=[ApimartError("GRSAI image task failed: detail=excessive system load"), succeeded],
                ) as edit_once,
                patch.object(grsai.time, "sleep") as sleep,
            ):
                result = editor.edit(
                    image,
                    "raw prompt",
                    "image edit prompt",
                    root / "output.png",
                    state_path,
                    aspect_ratio="16:9",
                )

            self.assertEqual(result, succeeded)
            self.assertEqual(edit_once.call_count, 2)
            sleep.assert_called_once_with(grsai.GRSAI_CAPACITY_RETRY_SECONDS)
            waiting_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(waiting_state["status"], "waiting_for_capacity")
            self.assertEqual(waiting_state["retry_after_seconds"], 60)

    def test_non_capacity_failure_is_raised_without_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            image.write_bytes(b"input")
            editor = grsai.GrsaiImageEditor.__new__(grsai.GrsaiImageEditor)
            error = ApimartError("GRSAI image task failed: invalid image")
            with (
                patch.object(editor, "_edit_once", side_effect=error),
                patch.object(grsai.time, "sleep") as sleep,
                self.assertRaisesRegex(ApimartError, "invalid image"),
            ):
                editor.edit(
                    image,
                    "raw prompt",
                    "image edit prompt",
                    root / "output.png",
                    root / "state.json",
                )
            sleep.assert_not_called()


class GrsaiModelPayloadTests(unittest.TestCase):
    def test_stage_model_routing(self) -> None:
        self.assertEqual(grsai.image_model_for_stage("S1"), "nano-banana-2")
        self.assertEqual(grsai.image_model_for_stage("s1"), "nano-banana-2")
        self.assertEqual(grsai.image_model_for_stage("S2"), "nano-banana-2")
        self.assertEqual(grsai.image_model_for_stage("S12"), "nano-banana-2")

    def test_edit_submits_nano_banana_two_and_downloads_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            output = root / "output.png"
            state = root / "state.json"
            Image.new("RGB", (16, 9), (12, 34, 56)).save(image)
            result_image = root / "result.png"
            Image.new("RGB", (16, 9), (56, 34, 12)).save(result_image)
            encoded = base64.b64encode(result_image.read_bytes()).decode("ascii")
            editor = grsai.GrsaiImageEditor.__new__(grsai.GrsaiImageEditor)
            editor.base_url = "https://grsaiapi.com"
            editor.timeout_seconds = 5
            responses = [
                {"status": "queued", "id": "task-1"},
                {"status": "succeeded", "url": f"data:image/png;base64,{encoded}"},
            ]
            with patch.object(editor, "request", side_effect=responses) as request:
                result = editor._edit_once(
                    image,
                    "make it warmer",
                    "Make it warmer. Preserve all other elements exactly as they are.",
                    output,
                    state,
                    aspect_ratio="16:9",
                )

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(request.call_args_list[0].args[:2], ("POST", "https://grsaiapi.com/v1/api/generate"))
            payload = request.call_args_list[0].args[2]
            self.assertEqual(payload["model"], grsai.GRSAI_IMAGE_MODEL)
            self.assertEqual(payload["model"], "nano-banana-2")
            self.assertEqual(payload["aspectRatio"], "16:9")
            self.assertEqual(result["model"], "nano-banana-2")
            with Image.open(output) as opened:
                self.assertEqual(opened.size, (16, 9))

    def test_edit_submits_nano_banana_two_for_every_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            output = root / "output.png"
            state = root / "state.json"
            Image.new("RGB", (16, 9), (12, 34, 56)).save(image)
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            editor = grsai.GrsaiImageEditor.__new__(grsai.GrsaiImageEditor)
            editor.base_url = "https://grsaiapi.com"
            editor.timeout_seconds = 5
            editor.model = grsai.image_model_for_stage("S2")
            with patch.object(
                editor,
                "request",
                return_value={"status": "succeeded", "url": f"data:image/png;base64,{encoded}"},
            ) as request:
                result = editor._edit_once(
                    image,
                    "replace the table",
                    "Replace the table. Preserve all other elements exactly as they are.",
                    output,
                    state,
                    aspect_ratio="16:9",
                )

            self.assertEqual(request.call_args.args[2]["model"], "nano-banana-2")
            self.assertEqual(result["model"], "nano-banana-2")
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["model"], "nano-banana-2")


if __name__ == "__main__":
    unittest.main()
