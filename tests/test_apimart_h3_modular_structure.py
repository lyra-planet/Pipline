from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_compatibility_entry_point_is_thin() -> None:
    entry = SCRIPTS / "run_apimart_minimax_h3_sequential.py"
    assert len(entry.read_text(encoding="utf-8").splitlines()) < 140


def test_all_pipeline_boundaries_import_without_provider_calls() -> None:
    modules = (
        "apimart_h3_pipeline.constants",
        "apimart_h3_pipeline.media",
        "apimart_h3_pipeline.policy",
        "apimart_h3_pipeline.vision",
        "apimart_h3_pipeline.image_editor",
        "apimart_h3_pipeline.bridge",
        "apimart_h3_pipeline.bridge_helpers",
        "apimart_h3_pipeline.bridge_execution",
        "apimart_h3_pipeline.artifacts",
        "apimart_h3_pipeline.cli",
        "apimart_h3_pipeline.prompt_catalog",
        "apimart_h3_pipeline.vision_client",
        "apimart_h3_pipeline.vision_refiner",
        "apimart_h3_pipeline.runner",
    )
    for name in modules:
        module = importlib.import_module(name)
        assert module.__file__ is not None


def test_temporal_contract_is_centralized() -> None:
    constants = importlib.import_module("apimart_h3_pipeline.constants")
    assert constants.QWEN_CONTEXT_FRAME_INDICES == (0, 26, 53, 80, 106)
    assert constants.PRIMARY_REFERENCE_FRAME_INDEX == 0
    assert constants.TEMPORAL_MIDDLE_FRAME_INDEX == 53
    assert constants.TEMPORAL_END_FRAME_INDEX == 106
    assert constants.REFERENCE_IMAGE_COUNTS == (1, 3)


def test_cli_credentials_are_machine_independent() -> None:
    runner = importlib.import_module("apimart_h3_pipeline.runner")
    argv = [
        "runner",
        "--compiled-jobs", "jobs.json",
        "--task-id", "t1",
        "--out-dir", "run",
        "--media-dir", "media",
        "--media-public-base-url", "https://media.invalid",
    ]
    with patch.dict(
        os.environ,
        {
            "APIMART_ENV_FILE": "/tmp/custom-apimart.env",
            "GRSAI_ENV_FILE": "/tmp/custom-grsai.env",
            "DASHSCOPE_ENV_FILE": "/tmp/custom-dashscope.env",
        },
        clear=False,
    ), patch.object(sys, "argv", argv):
        args = runner.parse_args()
    assert args.apimart_env == Path("/tmp/custom-apimart.env")
    assert args.grsai_env == Path("/tmp/custom-grsai.env")
    assert args.dashscope_env == Path("/tmp/custom-dashscope.env")


def test_console_entry_point_resolves_to_runner_main() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "apimart-h3-sequential = \"apimart_h3_pipeline.runner:main\"" in pyproject
