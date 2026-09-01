from __future__ import annotations

import json
from pathlib import Path

import pytest

from apimart_h3_pipeline.providers.local import LocalH3Client, LocalH3Config, LocalH3Error, LocalH3MediaAdapter


def template_graph() -> dict[str, dict[str, object]]:
    return {
        "load": {"class_type": "LoadVideo", "inputs": {"file": "old.mp4"}},
        "components": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "h3": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "prompt": "old",
                "width": 1,
                "height": 1,
                "length": 1,
                "ref_videos.ref_video_0": ["components", 0],
                "ref_images.ref_image_0": ["image", 0],
            },
        },
        "image": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "save": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "old"}},
    }


def client(tmp_path: Path, template: dict[str, object] | None = None) -> LocalH3Client:
    template_path = tmp_path / "workflow.json"
    template_path.write_text(json.dumps(template or template_graph()), encoding="utf-8")
    return LocalH3Client(LocalH3Config(
        server="http://127.0.0.1:8188",
        workflow_template=template_path,
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        timeout_seconds=2,
        poll_seconds=0.001,
    ))


def test_workflow_injection_discovers_nodes_and_clones_references(tmp_path: Path) -> None:
    graph = client(tmp_path).build_workflow(
        "Apply the edit.",
        "stage_input.mp4",
        ["start.png", "middle.png", "end.png"],
        "stage_S1",
    )
    h3 = graph["h3"]["inputs"]
    assert h3["prompt"] == "Apply the edit."
    assert (h3["width"], h3["height"], h3["length"]) == (1344, 768, 107)
    assert h3["ref_videos.ref_video_0"] == ["components", 0]
    assert h3["ref_images.ref_image_0"] == ["image", 0]
    assert h3["ref_images.ref_image_1"] != h3["ref_images.ref_image_0"]
    assert h3["ref_images.ref_image_2"] != h3["ref_images.ref_image_1"]
    assert graph[h3["ref_images.ref_image_1"][0]]["inputs"]["image"] == "middle.png"
    assert graph[h3["ref_images.ref_image_2"][0]]["inputs"]["image"] == "end.png"
    assert graph["load"]["inputs"]["file"] == "stage_input.mp4"
    assert graph["save"]["inputs"]["filename_prefix"] == "stage_S1"


def test_workflow_without_references_removes_image_nodes(tmp_path: Path) -> None:
    graph = client(tmp_path).build_workflow("Video only.", "input.mp4", [], "stage")
    assert all(node.get("class_type") != "LoadImage" for node in graph.values())
    assert not any(key.startswith("ref_images.") for key in graph["h3"]["inputs"])


def test_workflow_rejects_ambiguous_or_missing_node_contract(tmp_path: Path) -> None:
    bad = template_graph()
    bad["h3_copy"] = bad["h3"]
    with pytest.raises(LocalH3Error, match="exactly one MiniMax-H3"):
        client(tmp_path, bad).build_workflow("x", "input.mp4", [], "stage")

    missing = template_graph()
    del missing["save"]
    with pytest.raises(LocalH3Error, match="exactly one SaveVideo"):
        client(tmp_path, missing).build_workflow("x", "input.mp4", [], "stage")


def test_generate_materializes_files_and_resumes_saved_prompt(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"png")
    output = tmp_path / "stage" / "output.mp4"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    generated = output_dir / "stage_S1_00001.mp4"
    generated.write_bytes(b"generated")

    class FakeClient(LocalH3Client):
        def __init__(self, config: LocalH3Config) -> None:
            super().__init__(config)
            self.calls: list[tuple[str, object]] = []

        def _request_json(self, path: str, payload=None):  # type: ignore[no-untyped-def]
            self.calls.append((path, payload))
            if path == "/prompt":
                return {"prompt_id": "local-prompt"}
            return {
                "local-prompt": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"save": {"videos": [{"filename": generated.name, "subfolder": "", "type": "output"}]}},
                }
            }

    fake = FakeClient(client(tmp_path).config)
    result = fake.generate(
        source_video=source,
        prompt="Apply the edit.",
        reference_images=[reference],
        destination=output,
        stage_dir=tmp_path / "stage",
        stage_id="S1",
    )
    assert result == output
    assert output.read_bytes() == b"generated"
    assert [path for path, _ in fake.calls] == ["/prompt", "/history/local-prompt"]
    state = json.loads((tmp_path / "stage" / "local_task_state.json").read_text(encoding="utf-8"))
    assert state["prompt_id"] == "local-prompt"

    resumed = FakeClient(fake.config)
    resumed.generate(
        source_video=source,
        prompt="Apply the edit.",
        reference_images=[reference],
        destination=output,
        stage_dir=tmp_path / "stage",
        stage_id="S1",
    )
    assert [path for path, _ in resumed.calls] == ["/history/local-prompt"]


def test_local_media_adapter_returns_a_file_url(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")
    uploaded = LocalH3MediaAdapter.upload_image(image)
    assert uploaded["url"] == image.resolve().as_uri()
    assert uploaded["upload_mode"] == "local_file"
