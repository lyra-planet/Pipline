from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

SPEC = importlib.util.spec_from_file_location("vetra_failure_repair_test", SCRIPT_DIR / "vetra_failure_repair.py")
assert SPEC is not None and SPEC.loader is not None
policy_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy_module
SPEC.loader.exec_module(policy_module)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_apimart_minimax_h3_sequential_vetra_test",
    SCRIPT_DIR / "run_apimart_minimax_h3_sequential.py",
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


FailureDiagnosisAndRepair = policy_module.FailureDiagnosisAndRepair
RepairValidationError = policy_module.RepairValidationError
apply_repair_clause = policy_module.apply_repair_clause
fixed_three_anchor_repair = policy_module.fixed_three_anchor_repair
global_style_three_anchor_repair = policy_module.global_style_three_anchor_repair
stage_outcome = policy_module.stage_outcome
validate_observation = policy_module.validate_observation
validate_repair_record = policy_module.validate_repair_record
STAGE_RETRY_LIMIT = policy_module.STAGE_RETRY_LIMIT


def policy(needs_reference: bool = True, count: int = 1) -> dict[str, object]:
    return {"needs_reference_image": needs_reference, "reference_image_count": count}


def diagnosis(failure_type: str, confidence: float = 0.9, stage: str = "S2") -> dict[str, object]:
    return {
        "success": False,
        "failure_type": failure_type,
        "observation": "the requested edit was not confirmed",
        "observer_evidence": "the requested edit was not confirmed",
        "confidence": confidence,
        "stage_id": stage,
        "affected_scope": "current_stage_only",
        "repairable": True,
    }


class VetraObservationTests(unittest.TestCase):
    def test_success_defaults_unknown_type_to_none(self) -> None:
        result = validate_observation({"success": True, "observation": "confirmed", "confidence": 1.2})
        self.assertEqual(result["failure_type"], "none")
        self.assertEqual(result["confidence"], 1.0)

    def test_not_frame_judgeable_success_is_preserved(self) -> None:
        result = validate_observation({
            "success": True,
            "observation": "not_frame_judgeable",
            "confidence": 0.5,
        })
        self.assertEqual(result["failure_type"], "not_frame_judgeable")
        self.assertEqual(stage_outcome(result), "success")

    def test_null_observer_is_transport_failure(self) -> None:
        result = validate_observation({"success": None, "observation": "timeout", "confidence": -1})
        self.assertEqual(result["failure_type"], "observer_unavailable")
        self.assertEqual(stage_outcome(result), "observation_pending")
        self.assertEqual(stage_outcome(result, allow_unverified=True), "unverified_success")

    def test_invalid_success_and_failure_type_combination_is_rejected(self) -> None:
        with self.assertRaises(RepairValidationError):
            validate_observation({"success": True, "failure_type": "identity_drift", "observation": "drift"})
        with self.assertRaises(RepairValidationError):
            validate_observation({"success": False, "failure_type": "none", "observation": "missing"})

    def test_aliases_are_normalized(self) -> None:
        result = validate_observation({
            "success": False,
            "failure_type": "previous_edit_lost",
            "observation": "old edit disappeared",
        })
        self.assertEqual(result["failure_type"], "previous_stage_lost")


class VetraRepairPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = FailureDiagnosisAndRepair()
        self.current = "Move the red newspaper to the center of the table."
        self.failed = "Apply only this edit to <Video 1>: " + self.current

    def test_failure_to_action_and_one_anchor_mapping(self) -> None:
        expected = {
            "edit_missing": ("strengthen_edit", "one_anchor", 1),
            "identity_drift": ("strengthen_identity_preservation", "one_anchor", 1),
            "previous_stage_lost": ("strengthen_previous_stage_preservation", "one_anchor", 1),
            "style_inconsistency": ("use_three_anchor", "three_anchor", 3),
            "composition_weak": ("strengthen_composition", "one_anchor", 1),
        }
        for failure_type, (action, reference_policy, count) in expected.items():
            prior = [{"stage_id": "S1", "prompt": "Apply oil painting style.", "status": "confirmed"}]
            observed = diagnosis(failure_type)
            diagnosed = self.policy.diagnose(
                observed,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                previous_requirements=prior,
                evidence_frames=(3, 9),
            )
            repaired = self.policy.repair(
                diagnosed,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                previous_requirements=prior,
                retry_index=1,
                original_policy=policy(),
            )
            self.assertEqual(repaired["repair_action"], action)
            self.assertEqual(repaired["reference_policy"], reference_policy)
            self.assertEqual(repaired["reference_image_count"], count)
            self.assertIn("red", repaired["repaired_h3_prompt"])
            self.assertIn("newspaper", repaired["repaired_h3_prompt"])
            self.assertTrue(repaired["guard"]["same_stage"])

        self.assertEqual(diagnosed["evidence_frames"], [3, 9])

    def test_motion_weak_forces_video_only_without_picture(self) -> None:
        current = "Make a slight camera push-in toward the newspaper at a slow speed."
        observed = diagnosis("motion_weak")
        diagnosed = self.policy.diagnose(observed, stage_id="S2", current_requirement=current, failed_prompt=self.failed)
        repaired = self.policy.repair(
            diagnosed,
            stage_id="S2",
            current_requirement=current,
            failed_prompt=self.failed,
            retry_index=1,
            original_policy=policy(True, 1),
        )
        self.assertEqual(repaired["reference_policy"], "video_only")
        self.assertEqual(repaired["reference_image_count"], 0)
        self.assertNotIn("<Picture", repaired["repaired_h3_prompt"])
        self.assertIn("push-in", repaired["repaired_h3_prompt"])

    def test_motion_repair_requires_motion_semantics(self) -> None:
        observed = diagnosis("motion_weak")
        diagnosed = self.policy.diagnose(observed, stage_id="S2", current_requirement=self.current, failed_prompt=self.failed)
        with self.assertRaisesRegex(RepairValidationError, "motion cue"):
            self.policy.repair(
                diagnosed,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                retry_index=1,
                original_policy=policy(),
            )

    def test_previous_stage_loss_requires_confirmed_parent(self) -> None:
        observed = diagnosis("previous_stage_lost")
        diagnosed = self.policy.diagnose(observed, stage_id="S2", current_requirement=self.current, failed_prompt=self.failed)
        self.assertFalse(diagnosed["repairable"])
        with self.assertRaises(RepairValidationError):
            self.policy.repair(
                diagnosed,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                retry_index=1,
                original_policy=policy(),
            )

    def test_low_confidence_failure_is_not_repairable(self) -> None:
        diagnosed = self.policy.diagnose(
            diagnosis("edit_missing", confidence=0.2),
            stage_id="S2",
            current_requirement=self.current,
            failed_prompt=self.failed,
        )
        self.assertFalse(diagnosed["repairable"])

    def test_retry_budget_is_hard_capped(self) -> None:
        self.assertEqual(STAGE_RETRY_LIMIT, 1)
        with self.assertRaises(TypeError):
            FailureDiagnosisAndRepair(max_retries=2)
        diagnosed = self.policy.diagnose(
            diagnosis("edit_missing"), stage_id="S2", current_requirement=self.current, failed_prompt=self.failed,
        )
        with self.assertRaisesRegex(RepairValidationError, "repair budget"):
            self.policy.repair(
                diagnosed,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                retry_index=2,
                original_policy=policy(),
            )

        record = self.policy.repair(
            diagnosed,
            stage_id="S2",
            current_requirement=self.current,
            failed_prompt=self.failed,
            retry_index=1,
            original_policy=policy(),
        )
        record["max_retries"] = 2
        with self.assertRaisesRegex(RepairValidationError, "fixed retry limit"):
            validate_repair_record(record, stage_id="S2", current_requirement=self.current)

    def test_repaired_prompt_guard_rejects_dropped_requirement(self) -> None:
        with self.assertRaisesRegex(RepairValidationError, "dropped current requirement"):
            self.policy.validate_repaired_prompt(
                "Apply only this edit to <Video 1>: change the color. Make it clear.",
                self.current,
                "strengthen_edit",
            )

    def test_repaired_prompt_guard_rejects_structured_wrapper(self) -> None:
        with self.assertRaisesRegex(RepairValidationError, "structured task wrapper"):
            self.policy.validate_repaired_prompt(
                self.failed + " detailed_description: add another object.",
                self.current,
                "strengthen_edit",
            )

    def test_repair_rejects_stage_scope_change(self) -> None:
        bad = diagnosis("edit_missing")
        bad["affected_scope"] = "whole_plan"
        with self.assertRaisesRegex(RepairValidationError, "current_stage_only"):
            self.policy.repair(
                bad,
                stage_id="S2",
                current_requirement=self.current,
                failed_prompt=self.failed,
                retry_index=1,
                original_policy=policy(),
            )

    def test_video_only_original_policy_stays_video_only(self) -> None:
        diagnosed = self.policy.diagnose(
            diagnosis("composition_weak"), stage_id="S2", current_requirement=self.current, failed_prompt=self.failed,
        )
        repaired = self.policy.repair(
            diagnosed,
            stage_id="S2",
            current_requirement=self.current,
            failed_prompt=self.failed,
            retry_index=1,
            original_policy=policy(False, 0),
        )
        self.assertEqual(repaired["reference_policy"], "video_only")
        self.assertEqual(repaired["reference_image_count"], 0)

    def test_apply_clause_preserves_picture_contract(self) -> None:
        repaired = self.policy.repair(
            self.policy.diagnose(
                diagnosis("style_inconsistency"), stage_id="S2", current_requirement=self.current, failed_prompt=self.failed,
            ),
            stage_id="S2",
            current_requirement=self.current,
            failed_prompt=self.failed,
            retry_index=1,
            original_policy=policy(),
        )
        prompt = apply_repair_clause(
            "Use <Picture 1>, <Picture 2>, and <Picture 3> with <Video 1>.",
            current_requirement=self.current,
            repair=repaired,
            picture_count=3,
        )
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 3>", prompt)
        self.assertIn("appearance", prompt.lower())

    def test_fixed_three_anchor_mode_is_explicit(self) -> None:
        repaired = fixed_three_anchor_repair(
            stage_id="S2",
            current_requirement=self.current,
            failed_prompt=self.failed,
            observer_evidence="static edit absent",
        )
        self.assertEqual(repaired["mode"], "fixed_three_anchor")
        self.assertEqual(repaired["reference_image_count"], 3)
        self.assertFalse(repaired["guard"]["topology_changed"])

    def test_global_style_escalation_preserves_observed_failure_type(self) -> None:
        repaired = global_style_three_anchor_repair(
            stage_id="S1",
            current_requirement="Render the entire video as an oil painting.",
            failed_prompt="Apply <Video 1> oil painting.",
            observer_evidence="style appears only in the first frame",
            failure_type="style_inconsistency",
        )
        self.assertEqual(repaired["mode"], "global_style_three_anchor")
        self.assertEqual(repaired["failure_type"], "style_inconsistency")
        self.assertEqual(repaired["reference_policy"], "three_anchor")


class VetraPersistenceTests(unittest.TestCase):
    def test_repair_record_round_trip_validation(self) -> None:
        current = "Move the red newspaper to the center of the table."
        policy = FailureDiagnosisAndRepair()
        diagnosed = policy.diagnose(
            diagnosis("edit_missing"), stage_id="S2", current_requirement=current, failed_prompt="Apply <Video 1> " + current,
        )
        record = policy.repair(
            diagnosed,
            stage_id="S2",
            current_requirement=current,
            failed_prompt="Apply <Video 1> " + current,
            retry_index=1,
            original_policy={"needs_reference_image": True, "reference_image_count": 1},
        )
        normalized = validate_repair_record(record, stage_id="S2", current_requirement=current)
        self.assertEqual(normalized["repair_action"], "strengthen_edit")

    def test_persistence_rejects_changed_stage_or_reference_count(self) -> None:
        current = "Move the red newspaper to the center of the table."
        record = fixed_three_anchor_repair(
            stage_id="S2", current_requirement=current, failed_prompt="Apply <Video 1> " + current, observer_evidence="absent",
        )
        with self.assertRaises(RepairValidationError):
            validate_repair_record(record, stage_id="S3", current_requirement=current)
        altered = dict(record)
        altered["reference_image_count"] = 1
        with self.assertRaises(RepairValidationError):
            validate_repair_record(altered, stage_id="S2", current_requirement=current)

    def test_archived_attempts_are_sorted_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage_dir = Path(directory) / "S2"
            (stage_dir / "attempts" / "attempt_2").mkdir(parents=True)
            (stage_dir / "attempts" / "attempt_1").mkdir(parents=True)
            runner.write_json(stage_dir / "attempts" / "attempt_2" / "attempt.json", {"observation": {"success": False}})
            runner.write_json(stage_dir / "attempts" / "attempt_1" / "attempt.json", {"observation": {"success": False}})
            records = runner.load_archived_attempts(stage_dir)
        self.assertEqual([record["attempt"] for record in records], [1, 2])
        self.assertIn("post_edit_observation", records[0])

    def test_confirmed_previous_requirements_excludes_unconfirmed_stages(self) -> None:
        stages = [
            {"stage_id": "S1", "prompt": "Apply the watercolor style."},
            {"stage_id": "S2", "prompt": "Move the newspaper."},
            {"stage_id": "S3", "prompt": "Add a camera push-in."},
        ]
        manifest = {
            "stages": [
                {
                    "stage": "S1",
                    "raw_prompt": "Apply the watercolor style.",
                    "post_edit_observation": {"success": True},
                },
                {
                    "stage": "S2",
                    "raw_prompt": "Move the newspaper.",
                    "post_edit_observation": {"success": False},
                },
            ]
        }
        preserved = runner.confirmed_previous_requirements(manifest, stages, 2)
        self.assertEqual(preserved, [{
            "stage_id": "S1",
            "prompt": "Apply the watercolor style.",
            "status": "confirmed",
        }])


class VetraRunnerStateMachineTests(unittest.TestCase):
    def test_retry_limit_is_not_a_cli_option(self) -> None:
        argv = [
            "run_apimart_minimax_h3_sequential.py",
            "--compiled-jobs", "jobs.json",
            "--task-id", "t1",
            "--out-dir", "run",
            "--media-dir", "media",
            "--media-public-base-url", "https://media.invalid",
            "--max-stage-retries", "2",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            runner.parse_args()

    @staticmethod
    def _args(root: Path, recovery: str = "targeted") -> SimpleNamespace:
        return SimpleNamespace(
            compiled_jobs=root / "jobs.json",
            task_id="t1",
            out_dir=root / "run",
            media_dir=root / "media",
            media_public_base_url="https://media.invalid",
            prepared_initial_video=None,
            apimart_env=root / "apimart.env",
            grsai_env=root / "grsai.env",
            dashscope_env=root / "dashscope.env",
            dashscope_base_url="https://dashscope.invalid/v1",
            dashscope_model="qwen-vl-plus",
            dashscope_timeout=10,
            h3_model="MiniMax-H3",
            duration=4,
            resolution="768P",
            aspect_ratio="16:9",
            request_timeout=10,
            poll_seconds=0.01,
            total_timeout=10,
            allow_resubmit=False,
            last_stage=None,
            initial_reference=False,
            global_style_reference_count=1,
            failure_recovery=recovery,
            allow_unverified_output=False,
            dry_run=False,
        )

    @staticmethod
    def _fake_refiner(*args, **kwargs):
        return SimpleNamespace(model="qwen-vl-plus-test")

    @staticmethod
    def _fake_editor(*args, **kwargs):
        return SimpleNamespace()

    @staticmethod
    def _fake_apimart(*args, **kwargs):
        return SimpleNamespace(is_ctmoai=False, base_url="https://api.invalid")

    def _run_mocked_main(
        self,
        observations: list[dict[str, object]],
        recovery: str = "targeted",
        prompt: str = "Move the red newspaper to the center of the table.",
        stages: list[dict[str, str]] | None = None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            args = self._args(root, recovery=recovery)
            task = {
                "task_id": "t1",
                "source_video": source,
                "stages": stages or [{"stage_id": "S1", "prompt": prompt}],
            }
            geometry = runner.CanvasGeometry(1920, 1080, 1344, 756, 0, 6)
            bridge_calls: list[dict[str, object] | None] = []
            h3_calls: list[dict[str, str]] = []
            observed = iter(observations)

            def fake_bridge(*bridge_args, **bridge_kwargs):
                repair = bridge_kwargs.get("repair_context")
                if repair is None and len(bridge_args) >= 12:
                    repair = bridge_args[11]
                bridge_calls.append(dict(repair) if isinstance(repair, dict) else None)
                if isinstance(repair, dict):
                    count = int(repair.get("reference_image_count", 0))
                else:
                    count = int(runner.reference_policy(
                        str(bridge_args[5]), args.global_style_reference_count,
                    )["reference_image_count"])
                return [f"https://media.invalid/{index}" for index in range(count)], f"Apply only this edit to <Video 1>: {prompt}", {}

            def fake_h3(*call_args, **call_kwargs):
                stage_dir = Path(call_args[-1])
                stage_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / "output.mp4").write_bytes(f"output-{len(h3_calls)}".encode())
                h3_calls.append({
                    "prompt": str(call_args[2]),
                    "video_url": str(call_args[3]),
                })

            def fake_observe(*args, **kwargs):
                return next(observed)

            def fake_initial(source_path, target, geometry_value):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"initial")
                return target

            def fake_stage(source_path, target, geometry_value):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(source_path).read_bytes())
                return target

            def fake_final(source_path, target, geometry_value):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(source_path).read_bytes())
                return target

            patches = [
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "load_task", return_value=task),
                patch.object(runner, "source_canvas_geometry", return_value=geometry),
                patch.object(runner, "materialize_initial_video", side_effect=fake_initial),
                patch.object(runner, "materialize_stage_video", side_effect=fake_stage),
                patch.object(runner, "materialize_final_video", side_effect=fake_final),
                patch.object(runner, "resolve_credentials", return_value=("key", "https://api.invalid")),
                patch.object(runner, "ApimartClient", side_effect=self._fake_apimart),
                patch.object(runner, "DashScopeVisionRefiner", side_effect=self._fake_refiner),
                patch.object(runner, "GrsaiImageEditor", side_effect=self._fake_editor),
                patch.object(runner, "bridge_for_stage", side_effect=fake_bridge),
                patch.object(runner, "invoke_h3_client", side_effect=fake_h3),
                patch.object(runner, "observe_stage_output", side_effect=fake_observe),
                patch.object(runner, "is_aligned_video", side_effect=lambda path: Path(path).is_file()),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], patches[12], patches[13]:
                error = None
                try:
                    result = runner.main()
                except Exception as exc:  # Assert the exact error in the caller.
                    result = None
                    error = exc
            manifest_path = args.out_dir / "sequence_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            return result, error, bridge_calls, h3_calls, manifest

    def test_targeted_retry_is_propagated_without_a_second_observer(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": False, "failure_type": "edit_missing", "observation": "edit absent", "confidence": 0.95},
            {"success": True, "failure_type": "none", "observation": "unused", "confidence": 0.95},
        ])
        self.assertIsNone(error)
        self.assertEqual(result, 0)
        self.assertEqual(len(h3_calls), 2)
        self.assertIsNone(bridge_calls[0])
        self.assertEqual(bridge_calls[1]["repair_action"], "strengthen_edit")
        self.assertEqual(bridge_calls[1]["reference_policy"], "one_anchor")
        self.assertEqual(manifest["stages"][0]["status"], "semantic_failure_propagated")
        self.assertEqual(manifest["status"], "degraded")
        self.assertEqual(len(manifest["stages"][0]["attempts"]), 2)
        self.assertEqual(manifest["stages"][0]["attempts"][0]["diagnosis"]["failure_type"], "edit_missing")
        self.assertTrue(manifest["stages"][0]["attempts"][1]["observer_skipped"])

    def test_single_retry_reuses_stage_parent_and_does_not_stop_sequence(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": False, "failure_type": "edit_missing", "observation": "edit absent", "confidence": 0.95},
            {"success": False, "failure_type": "edit_missing", "observation": "still absent", "confidence": 0.95},
            {"success": True, "failure_type": "none", "observation": "confirmed", "confidence": 0.95},
        ])
        self.assertEqual(result, 0)
        self.assertIsNone(error)
        self.assertEqual(len(h3_calls), 2)
        self.assertEqual(
            [call["video_url"] for call in h3_calls],
            ["https://media.invalid/task_t1_initial.mp4"] * 2,
        )
        self.assertEqual(
            [attempt["h3_input_video_url"] for attempt in manifest["stages"][0]["attempts"]],
            ["https://media.invalid/task_t1_initial.mp4"] * 2,
        )
        self.assertEqual(manifest["status"], "degraded")
        self.assertEqual(manifest["stages"][0]["status"], "semantic_failure_propagated")

    def test_retry_output_is_propagated_even_when_retry_would_have_failed(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": False, "failure_type": "identity_drift", "observation": "face changed", "confidence": 0.95},
            {"success": False, "failure_type": "identity_drift", "observation": "face still changed", "confidence": 0.95},
        ])
        self.assertEqual(result, 0)
        self.assertIsNone(error)
        self.assertEqual(len(h3_calls), 2)
        self.assertEqual(manifest["status"], "degraded")
        self.assertEqual(manifest["stages"][0]["status"], "semantic_failure_propagated")
        self.assertEqual(manifest["stages"][0]["diagnosis"]["failure_type"], "identity_drift")
        self.assertTrue(manifest["stages"][0]["observer_skipped"])

    def test_retry_output_becomes_next_stage_parent(self) -> None:
        result, error, _bridge_calls, h3_calls, manifest = self._run_mocked_main(
            [
                {"success": False, "failure_type": "edit_missing", "observation": "absent", "confidence": 0.95},
                {"success": True, "failure_type": "none", "observation": "stage two unused", "confidence": 0.95},
            ],
            stages=[
                {"stage_id": "S1", "prompt": "Move the red newspaper to the center of the table."},
                {"stage_id": "S2", "prompt": "Add a blue cup beside the newspaper."},
            ],
        )
        self.assertEqual(result, 0)
        self.assertIsNone(error)
        self.assertEqual(len(h3_calls), 3)
        self.assertEqual(h3_calls[0]["video_url"], "https://media.invalid/task_t1_initial.mp4")
        self.assertEqual(h3_calls[1]["video_url"], "https://media.invalid/task_t1_initial.mp4")
        self.assertEqual(h3_calls[2]["video_url"], "https://media.invalid/task_t1_S1.mp4")
        self.assertEqual(manifest["stages"][0]["status"], "semantic_failure_propagated")
        self.assertEqual(manifest["stages"][1]["status"], "success")

    def test_observer_unavailable_stops_without_propagation(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": None, "observation": "timeout", "confidence": 0.0},
        ])
        self.assertIsNone(result)
        self.assertIsInstance(error, runner.ApimartError)
        self.assertEqual(len(h3_calls), 1)
        self.assertEqual(manifest["status"], "observation_pending")
        self.assertEqual(manifest["stages"][0]["status"], "observation_pending")
        self.assertIsNone(manifest["stages"][0]["repair"])

    def test_disabled_recovery_stops_after_first_semantic_failure(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": False, "failure_type": "edit_missing", "observation": "absent", "confidence": 0.95},
        ], recovery="disabled")
        self.assertIsNone(result)
        self.assertIsInstance(error, runner.ApimartError)
        self.assertEqual(len(h3_calls), 1)
        self.assertEqual(manifest["status"], "semantic_failure")
        self.assertEqual(manifest["stages"][0]["status"], "semantic_failure")
        self.assertIsNone(bridge_calls[0])

    def test_fixed_mode_keeps_legacy_three_anchor_fallback(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main([
            {"success": False, "failure_type": "edit_missing", "observation": "edit absent", "confidence": 0.95},
            {"success": True, "failure_type": "none", "observation": "confirmed", "confidence": 0.95},
        ], recovery="fixed-three-anchor")
        self.assertIsNone(error)
        self.assertEqual(result, 0)
        self.assertEqual(len(h3_calls), 2)
        self.assertEqual(bridge_calls[1]["mode"], "fixed_three_anchor")
        self.assertEqual(bridge_calls[1]["reference_image_count"], 3)
        self.assertEqual(manifest["stages"][0]["reference_escalated"], True)

    def test_global_style_failure_directly_uses_three_anchor_targeted_retry(self) -> None:
        result, error, bridge_calls, h3_calls, manifest = self._run_mocked_main(
            [
                {"success": False, "failure_type": "unclassified", "observation": "style is not stable", "confidence": 0.95},
                {"success": True, "failure_type": "none", "observation": "confirmed", "confidence": 0.95},
            ],
            prompt="Render the entire video as an oil painting.",
        )
        self.assertIsNone(error)
        self.assertEqual(result, 0)
        self.assertEqual(len(h3_calls), 2)
        self.assertIsNone(bridge_calls[0])
        self.assertEqual(bridge_calls[1]["mode"], "global_style_three_anchor")
        self.assertEqual(bridge_calls[1]["reference_image_count"], 3)
        self.assertEqual(manifest["stages"][0]["attempts"][0]["reference_image_count"], 1)
        self.assertEqual(manifest["stages"][0]["attempts"][1]["reference_image_count"], 3)

    def test_deterministic_three_anchor_prompt_contains_temporal_roles(self) -> None:
        repair = fixed_three_anchor_repair(
            stage_id="S2",
            current_requirement="Render the entire video as an oil painting.",
            failed_prompt="Apply <Video 1> oil painting.",
            observer_evidence="style changed over time",
        )
        prompt = runner.deterministic_repair_h3_prompt(
            "Render the entire video as an oil painting.",
            3,
            [
                {"role": "edited start anchor", "source_frame_index": 0},
                {"role": "edited primary anchor", "source_frame_index": 53},
                {"role": "edited end anchor", "source_frame_index": 106},
            ],
            repair,
        )
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("source frame 53", prompt)
        self.assertIn("edited end anchor", prompt)
        self.assertIn("oil painting", prompt)

    def test_three_anchor_retry_reuses_archived_primary_anchor(self) -> None:
        """A retry must not regenerate the primary image when the first bridge used 3 anchors."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.mp4"
            parent.write_bytes(b"test video placeholder")
            stage_dir = root / "S2"
            class Refiner:
                model = "qwen-vl-plus-test"

                def __init__(self) -> None:
                    self.plan_calls = 0
                    self.fallback_calls = 0
                    self.compose_calls = 0

                def plan_reference(self, frames, prompt, is_global_style):
                    self.plan_calls += 1
                    return {
                        "model": self.model,
                        "selected_frame_index": 0,
                        "selection_reason": "clear frame",
                        "image_edit_prompt": prompt,
                        "frame_observation": "clear",
                        "is_global_style": is_global_style,
                        "usage": {},
                    }

                def plan_fallback_anchors(self, *args, **kwargs):
                    self.fallback_calls += 1
                    return {
                        "model": self.model,
                        "selected_frame_index": 0,
                        "middle_image_edit_prompt": args[2],
                        "end_image_edit_prompt": args[2],
                        "frame_observation": "clear",
                        "is_global_style": args[4],
                        "usage": {},
                    }

                def compose_h3_prompt(self, frames, references, prompt, is_global_style, reference_roles=(), failure_observation=None):
                    self.compose_calls += 1
                    tags = " ".join(f"<Picture {i}>" for i in range(1, len(references) + 1))
                    roles = " ".join(
                        f"<Picture {i}> = {role['role']}, source frame {role['source_frame_index']}"
                        for i, role in enumerate(reference_roles, 1)
                    )
                    return {
                        "model": self.model,
                        "h3_prompt": f"{tags} {roles} <Video 1> Apply only this edit.".strip(),
                        "frame_observation": "clear",
                        "picture_count": len(references),
                        "is_global_style": is_global_style,
                        "usage": {},
                    }

            class Editor:
                def __init__(self) -> None:
                    self.calls = []

                def edit(self, image, raw_prompt, image_edit_prompt, output, state_path, style_reference=None):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"png")
                    self.calls.append({"output": str(output), "style_reference": style_reference})
                    return {"status": "succeeded", "output": str(output)}

            class Ctmoai:
                is_ctmoai = False

            qwen = Refiner()
            editor = Editor()
            def fake_frame(video, output, frame_index=53):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"frame-{frame_index}".encode())

            with patch.object(runner, "select_keyframe", fake_frame):
                first_urls, _, _ = runner.bridge_for_stage(
                    qwen,
                    editor,
                    Ctmoai(),
                    parent,
                    stage_dir,
                    "Recolor the background blue.",
                    3,
                    "https://media.invalid",
                    root / "media",
                    "139",
                )
                self.assertEqual(len(first_urls), 3)
                archive = stage_dir / "attempts" / "attempt_1"
                archive.mkdir(parents=True)
                shutil.move(str(stage_dir / "bridge_for_next"), archive / "bridge_for_next")
                repair = runner.fixed_three_anchor_repair(
                    stage_id="S2",
                    current_requirement="Recolor the background blue.",
                    failed_prompt="Apply <Video 1> recolor the background blue.",
                    observer_evidence="style inconsistent",
                )
                second_urls, _, second_bridge = runner.bridge_for_stage(
                    qwen,
                    editor,
                    Ctmoai(),
                    parent,
                    stage_dir,
                    "Recolor the background blue.",
                    3,
                    "https://media.invalid",
                    root / "media",
                    "139",
                    "style inconsistent",
                    repair,
                )

            self.assertEqual(len(second_urls), 3)
            self.assertEqual(qwen.plan_calls, 1)
            self.assertEqual(len(editor.calls), 5)
            self.assertEqual(second_bridge["image_edits"][0]["status"], "reused_from_attempt_1")


if __name__ == "__main__":
    unittest.main()
