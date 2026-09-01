from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_apimart_minimax_h3_sequential.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_apimart_minimax_h3_sequential_multiframe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class FakeRefiner:
    model = "qwen-vl-plus-test"

    def __init__(self) -> None:
        self.plan_calls = 0
        self.compositions: list[list[str]] = []
        self.composition_roles: list[list[dict[str, object]]] = []
        self.composition_failure_observations: list[str | None] = []

    def plan_reference(self, context_frames, raw_prompt, is_global_style):
        self.plan_calls += 1
        self._check_context(context_frames)
        return {
            "model": self.model,
            "selected_frame_index": 0,
            "selection_reason": "frame 0 is the deterministic visual style master",
            "image_edit_prompt": "Apply the requested watercolor style while preserving the scene.",
            "frame_observation": "target visible",
            "is_global_style": is_global_style,
            "usage": {},
        }

    def compose_h3_prompt(
        self, context_frames, references, raw_prompt, is_global_style,
        reference_roles=(), failure_observation=None,
    ):
        self._check_context(context_frames)
        self.compositions.append([Path(item).name for item in references])
        self.composition_roles.append([dict(role) for role in reference_roles])
        self.composition_failure_observations.append(failure_observation)
        picture_tags = " ".join(f"<Picture {index}>" for index in range(1, len(references) + 1))
        role_text = " ".join(
            f"<Picture {index}> = {role['role']}, source frame {role['source_frame_index']}"
            for index, role in enumerate(reference_roles, 1)
        )
        return {
            "model": self.model,
            "h3_prompt": f"{picture_tags} {role_text} <Video 1> Apply only the requested edit.".strip(),
            "frame_observation": "inputs inspected",
            "picture_count": len(references),
            "is_global_style": is_global_style,
            "usage": {},
        }

    @staticmethod
    def _check_context(context_frames) -> None:
        assert len(context_frames) == 5
        assert [Path(item).stem.rsplit("_", 1)[-1] for item in context_frames] == [
            "000", "026", "053", "080", "106",
        ]


class FakeEditor:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def edit(
        self, image, raw_prompt, image_edit_prompt, output, state_path, style_reference=None,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output)
        call = {
            "image": str(image),
            "raw_prompt": raw_prompt,
            "image_edit_prompt": image_edit_prompt,
            "output": str(output),
            "style_reference": str(style_reference) if style_reference else None,
        }
        self.calls.append(call)
        return {"status": "succeeded", **call}


class FakeCtmoai:
    is_ctmoai = True

    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload_media(self, image, media_type):
        assert media_type == "images"
        self.uploads.append(Path(image).name)
        return f"https://media.invalid/{Path(image).name}"


def fake_select_keyframe(video: Path, output: Path, frame_index: int = 53) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 18), (frame_index, 20, 30)).save(output, "PNG")


class H3MultiframeBridgeTests(unittest.TestCase):
    def test_qwen_reference_planner_requires_first_parent_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context_frames = []
            for frame_index in pipeline.QWEN_CONTEXT_FRAME_INDICES:
                frame = root / f"context_{frame_index:03d}.png"
                Image.new("RGB", (32, 18), (frame_index, 20, 30)).save(frame, "PNG")
                context_frames.append(frame)

            refiner = pipeline.DashScopeVisionRefiner.__new__(pipeline.DashScopeVisionRefiner)
            refiner.model = "qwen-vl-plus-test"
            captured_payloads = []

            def complete_success(payload):
                captured_payloads.append(payload)
                return (
                    json.dumps({
                        "selected_frame_index": pipeline.PRIMARY_REFERENCE_FRAME_INDEX,
                        "selection_reason": "first frame is the style master",
                        "frame_observation": "all five frames inspected",
                    }),
                    {"usage": {}},
                )

            refiner.complete = complete_success
            result = refiner.plan_reference(
                context_frames,
                "Transform the entire video into a watercolor style.",
                True,
            )
            self.assertEqual(result["selected_frame_index"], pipeline.PRIMARY_REFERENCE_FRAME_INDEX)
            self.assertIn(
                f"must be the integer {pipeline.PRIMARY_REFERENCE_FRAME_INDEX}",
                captured_payloads[0]["messages"][0]["content"],
            )

            def complete_middle(payload):
                return (
                    json.dumps({
                        "selected_frame_index": pipeline.TEMPORAL_MIDDLE_FRAME_INDEX,
                        "selection_reason": "middle frame is clear",
                        "frame_observation": "all five frames inspected",
                    }),
                    {"usage": {}},
                )

            refiner.complete = complete_middle
            with self.assertRaises(pipeline.ApimartError):
                refiner.plan_reference(
                    context_frames,
                    "Transform the entire video into a watercolor style.",
                    True,
                )

    def test_failed_single_reference_reuses_qwen_frame_as_three_anchor_master(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S2"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Recolor the background blue."

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                image_urls, _, first_bridge = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 1, task_id="139",
                )

                self.assertEqual(len(image_urls), 1)
                self.assertEqual(first_bridge["selected_frame_index"], 0)
                self.assertTrue(editor.calls[0]["image"].endswith("_S2_context_frame_000.png"))
                self.assertIsNone(editor.calls[0]["style_reference"])

                attempt_dir = stage_dir / "attempts" / "attempt_1"
                attempt_dir.mkdir(parents=True)
                shutil.move(str(stage_dir / "bridge_for_next"), attempt_dir / "bridge_for_next")

                image_urls, h3_prompt, fallback_bridge = pipeline.bridge_for_stage(
                    refiner,
                    editor,
                    apimart,
                    parent,
                    stage_dir,
                    raw_prompt,
                    3,
                    task_id="139",
                    failure_observation="the requested style was absent in the output",
                )

            self.assertEqual(refiner.plan_calls, 1)
            self.assertEqual(len(image_urls), 3)
            self.assertEqual(
                [Path(item).name.rsplit("_", 1)[-1] for item in fallback_bridge["reference_images"]],
                ["000.png", "053.png", "106.png"],
            )
            self.assertEqual(len(editor.calls), 3)
            self.assertIsNone(editor.calls[0]["style_reference"])
            self.assertIsNotNone(editor.calls[1]["style_reference"])
            self.assertIsNotNone(editor.calls[2]["style_reference"])
            self.assertTrue(editor.calls[1]["image"].endswith("_S2_context_frame_053.png"))
            self.assertTrue(editor.calls[2]["image"].endswith("_S2_context_frame_106.png"))
            self.assertEqual(editor.calls[1]["image_edit_prompt"], raw_prompt)
            self.assertEqual(editor.calls[2]["image_edit_prompt"], raw_prompt)
            self.assertTrue(str(editor.calls[1]["style_reference"]).endswith("_S2_reference_frame_000.png"))
            self.assertEqual(editor.calls[1]["style_reference"], editor.calls[2]["style_reference"])
            self.assertIn("<Picture 1>", h3_prompt)
            self.assertIn("<Picture 2>", h3_prompt)
            self.assertIn("<Picture 3>", h3_prompt)
            self.assertEqual(refiner.compositions[-1], [
                "task_139_S2_reference_frame_000.png",
                "task_139_S2_reference_frame_053.png",
                "task_139_S2_reference_frame_106.png",
            ])
            self.assertEqual(refiner.composition_roles[-1], [
                {"picture_index": 1, "role": "edited start anchor", "source_frame_index": 0},
                {"picture_index": 2, "role": "edited primary anchor", "source_frame_index": 53},
                {"picture_index": 3, "role": "edited end anchor", "source_frame_index": 106},
            ])
            self.assertEqual(
                refiner.composition_failure_observations[-1],
                "the requested style was absent in the output",
            )
            self.assertEqual(fallback_bridge["reference_roles"], refiner.composition_roles[-1])
            self.assertEqual(
                fallback_bridge["failure_observation"],
                "the requested style was absent in the output",
            )

    def test_global_style_defaults_to_one_anchor_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S1"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Transform the entire video into a watercolor style."

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                image_urls, h3_prompt, bridge = pipeline.bridge_for_stage(
                    refiner,
                    editor,
                    apimart,
                    parent,
                    stage_dir,
                    raw_prompt,
                    global_style_reference_count=1,
                    task_id="style",
                )

            self.assertEqual(len(image_urls), 1)
            self.assertEqual(bridge["policy"]["reference_image_count"], 1)
            self.assertEqual(bridge["policy"]["policy_reason"], "global_visual_style_change")
            self.assertTrue(editor.calls[0]["image"].endswith("_S1_context_frame_000.png"))
            self.assertIn("<Picture 1>", h3_prompt)
            self.assertEqual(len(refiner.compositions), 1)

    def test_global_style_three_anchor_contract_is_used_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S1"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Transform the entire video into a watercolor style."
            repair = pipeline.global_style_three_anchor_repair(
                stage_id="S1",
                current_requirement=raw_prompt,
                failed_prompt="Apply only this edit to <Video 1>: " + raw_prompt,
                observer_evidence="the style is inconsistent across time",
                failure_type="style_inconsistency",
            )

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                first_urls, _, _ = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 1, task_id="style",
                )
                self.assertEqual(len(first_urls), 1)
                attempt_dir = stage_dir / "attempts" / "attempt_1"
                attempt_dir.mkdir(parents=True)
                shutil.move(str(stage_dir / "bridge_for_next"), attempt_dir / "bridge_for_next")
                retry_urls, h3_prompt, bridge = pipeline.bridge_for_stage(
                    refiner,
                    editor,
                    apimart,
                    parent,
                    stage_dir,
                    raw_prompt,
                    1,
                    task_id="style",
                    repair_context=repair,
                )

            self.assertEqual(len(retry_urls), pipeline.GLOBAL_STYLE_REFERENCE_COUNT)
            self.assertEqual(bridge["policy"]["reference_image_count"], pipeline.GLOBAL_STYLE_REFERENCE_COUNT)
            self.assertEqual(bridge["policy"]["policy_reason"], "vetra_repair:use_three_anchor")
            self.assertIn("edited start anchor at source frame 0", h3_prompt)
            self.assertIn("edited primary anchor at source frame 53", h3_prompt)
            self.assertIn("edited end anchor at source frame 106", h3_prompt)
            self.assertIn("Frame-lock the edited appearance", h3_prompt)
            self.assertIn("Maintain this edited appearance in every frame", h3_prompt)
            self.assertIn("never revert to the source appearance", h3_prompt)
            self.assertEqual(bridge["final_refiner"]["h3_prompt_source"], "vetra_deterministic_repair")
            self.assertEqual(len(refiner.compositions), 1)

    def test_static_three_anchor_override_does_not_change_global_initial_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S1"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Transform the entire video into a watercolor style."

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                image_urls, h3_prompt, bridge = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 3, task_id="style",
                )

            self.assertEqual(len(image_urls), 1)
            self.assertIn("<Picture 1>", h3_prompt)
            self.assertEqual(bridge["policy"]["reference_image_count"], 1)
            self.assertEqual(len(refiner.compositions), 1)

    def test_local_static_edit_keeps_one_anchor_by_default(self) -> None:
        policy = pipeline.reference_policy("Recolor the background blue.")
        self.assertFalse(policy["is_global_style"])
        self.assertEqual(policy["reference_image_count"], 1)

    def test_three_anchor_plan_declares_first_frame_style_master(self) -> None:
        plan = pipeline.three_anchor_reference_plan(
            "qwen-vl-plus-test",
            "Transform the entire video into a watercolor style.",
            pipeline.PRIMARY_REFERENCE_FRAME_INDEX,
            True,
        )
        self.assertEqual(plan["selected_frame_index"], pipeline.PRIMARY_REFERENCE_FRAME_INDEX)
        self.assertEqual(plan["style_reference_frame_index"], pipeline.PRIMARY_REFERENCE_FRAME_INDEX)
        self.assertEqual(plan["middle_frame_index"], pipeline.TEMPORAL_MIDDLE_FRAME_INDEX)
        self.assertEqual(plan["end_frame_index"], pipeline.TEMPORAL_END_FRAME_INDEX)
        self.assertEqual(plan["middle_image_edit_prompt"], "Transform the entire video into a watercolor style.")
        self.assertNotIn("start_image_edit_prompt", plan)
        with self.assertRaises(pipeline.ApimartError):
            pipeline.three_anchor_reference_plan(
                "qwen-vl-plus-test",
                "Transform the entire video into a watercolor style.",
                pipeline.TEMPORAL_MIDDLE_FRAME_INDEX,
                True,
            )

    def test_legacy_middle_frame_roles_are_not_reused_from_bridge_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S2"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Recolor the background blue."

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                _, _, bridge = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 3, task_id="139",
                )
                self.assertTrue(pipeline.reference_roles_match_policy(bridge["reference_roles"], 3))
                stale_bridge = pipeline.read_json(stage_dir / "bridge_for_next" / "bridge.json")
                stale_bridge["reference_roles"][1]["source_frame_index"] = pipeline.OBSERVATION_FRAME_INDICES[-2]
                pipeline.write_json(stage_dir / "bridge_for_next" / "bridge.json", stale_bridge)
                _, _, regenerated = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 3, task_id="139",
                )

            self.assertFalse(
                pipeline.reference_roles_match_policy(stale_bridge["reference_roles"], 3)
            )
            self.assertEqual(regenerated["reference_roles"], pipeline.expected_reference_roles(3))
            self.assertEqual(len(editor.calls), 6)

    def test_video_only_stage_still_uses_five_frames_for_qwen_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                image_urls, h3_prompt, bridge = pipeline.bridge_for_stage(
                    refiner,
                    editor,
                    apimart,
                    parent,
                    root / "S3",
                    "Add a gentle camera pull-back.",
                    1,
                    task_id="139",
                )

            self.assertEqual(image_urls, [])
            self.assertEqual(editor.calls, [])
            self.assertEqual(refiner.plan_calls, 0)
            self.assertEqual(refiner.compositions, [[]])
            self.assertEqual(bridge["context_frame_indices"], [0, 26, 53, 80, 106])
            self.assertIn("<Video 1>", h3_prompt)
            self.assertNotIn("<Picture", h3_prompt)

    def test_targeted_repair_reuses_primary_reference_without_qwen_recompose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S2"
            refiner = FakeRefiner()
            editor = FakeEditor()
            apimart = FakeCtmoai()
            raw_prompt = "Replace the newspaper with a blue newspaper while preserving the person."
            repair_policy = pipeline.FailureDiagnosisAndRepair()
            diagnosis = {
                "success": False,
                "failure_type": "identity_drift",
                "observer_evidence": "the person's face changed",
                "observation": "the person's face changed",
                "confidence": 0.95,
                "stage_id": "S2",
                "affected_scope": "current_stage_only",
                "repairable": True,
            }
            repair = repair_policy.repair(
                diagnosis,
                stage_id="S2",
                current_requirement=raw_prompt,
                failed_prompt="Apply only this edit to <Video 1>: " + raw_prompt,
                retry_index=1,
                original_policy={"needs_reference_image": True, "reference_image_count": 1},
            )

            with patch.object(pipeline, "select_keyframe", fake_select_keyframe):
                first_urls, _, _ = pipeline.bridge_for_stage(
                    refiner, editor, apimart, parent, stage_dir, raw_prompt, 1, task_id="139",
                )
                self.assertEqual(len(first_urls), 1)
                attempt_dir = stage_dir / "attempts" / "attempt_1"
                attempt_dir.mkdir(parents=True)
                shutil.move(str(stage_dir / "bridge_for_next"), attempt_dir / "bridge_for_next")
                repaired_urls, repaired_prompt, repaired_bridge = pipeline.bridge_for_stage(
                    refiner,
                    editor,
                    apimart,
                    parent,
                    stage_dir,
                    raw_prompt,
                    1,
                    task_id="139",
                    failure_observation="the person's face changed",
                    repair_context=repair,
                )

            self.assertEqual(len(repaired_urls), 1)
            self.assertEqual(refiner.plan_calls, 1)
            self.assertEqual(len(editor.calls), 1)
            self.assertEqual(refiner.compositions, [["task_139_S2_reference_frame_000.png"]])
            self.assertIn("identity", repaired_prompt.lower())
            self.assertEqual(repaired_bridge["repair_context"]["repair_action"], "strengthen_identity_preservation")


if __name__ == "__main__":
    unittest.main()
