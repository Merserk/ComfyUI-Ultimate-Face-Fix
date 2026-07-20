from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WORKFLOWS = {
    "gen_to_img_face_fix_SDXL.json",
    "img_to_img_face_fix_SDXL.json",
    "gen_to_img_face_fix_KREA_2_Turbo.json",
    "img_to_img_face_fix_KREA_2_Turbo.json",
}


def _load_workflow(name: str) -> dict:
    workflow = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    UUID(workflow["id"])
    assert workflow["version"] == 0.4
    assert workflow["nodes"]
    return workflow


def _validate_graph(workflow: dict) -> dict[str, dict]:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    links = {link[0]: link for link in workflow["links"]}
    assert len(links) == len(workflow["links"])
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert origin_id in nodes and target_id in nodes
        assert nodes[origin_id]["outputs"][origin_slot]["type"] == link_type
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
    return {node["type"]: node for node in workflow["nodes"]}


def _linked_target(workflow: dict, node: dict, output_slot: int) -> dict:
    link_id = node["outputs"][output_slot]["links"][0]
    link = next(link for link in workflow["links"] if link[0] == link_id)
    return next(candidate for candidate in workflow["nodes"] if candidate["id"] == link[3])


def test_only_the_four_supported_workflows_are_shipped():
    assert {path.name for path in EXAMPLES.glob("*.json")} == WORKFLOWS
    assert not (EXAMPLES / "single_face.json").exists()
    assert not (EXAMPLES / "multi_face.json").exists()


def test_all_workflows_use_multi_face_defaults_and_crop_preview():
    for name in WORKFLOWS:
        workflow = _load_workflow(name)
        by_type = _validate_graph(workflow)
        assert {
            "UltimateFaceFixDetectorLoader",
            "UltimateFaceFixParserLoader",
            "LoadMediaPipeFaceLandmarker",
            "UltimateFaceFix",
            "SaveImage",
        } <= by_type.keys()
        settings = by_type["UltimateFaceFix"]["widgets_values"]
        assert settings[0] == "all"
        assert settings[1] == "maximum"
        assert settings[2] == 8
        assert settings[6] == "768"
        assert settings[7] == "repair"
        assert settings[13:15] == ["euler", "beta"]
        assert by_type["UltimateFaceFixDetectorLoader"]["widgets_values"] == ["face_yolov9c.pt"]
        assert by_type["UltimateFaceFixParserLoader"]["widgets_values"] == [
            "segface_convnext_celeba_512.safetensors"
        ]
        assert by_type["LoadMediaPipeFaceLandmarker"]["widgets_values"] == [
            "mediapipe_face_fp32.safetensors"
        ]
        crop_preview = _linked_target(workflow, by_type["UltimateFaceFix"], 2)
        assert crop_preview["type"] == "PreviewImage"
        assert crop_preview["title"] == "PROCESSED FACE CROPS"


def test_sdxl_workflows_have_blank_checkpoint_and_expected_source_path():
    for prefix in ("gen", "img"):
        workflow = _load_workflow(f"{prefix}_to_img_face_fix_SDXL.json")
        by_type = _validate_graph(workflow)
        assert by_type["CheckpointLoaderSimple"]["widgets_values"] == [""]
        source_type = "VAEDecode" if prefix == "gen" else "LoadImage"
        assert by_type["UltimateFaceFix"]["inputs"][0]["link"] in by_type[source_type]["outputs"][0]["links"]
        if prefix == "gen":
            assert {"EmptyLatentImage", "KSampler", "VAEDecode"} <= by_type.keys()
            assert by_type["KSampler"]["widgets_values"][2:6] == [25, 5.5, "euler", "beta"]
        else:
            assert by_type["LoadImage"]["widgets_values"][0] == ""


def test_krea_workflows_use_native_blank_loader_stack_and_zero_negative():
    for prefix in ("gen", "img"):
        workflow = _load_workflow(f"{prefix}_to_img_face_fix_KREA_2_Turbo.json")
        by_type = _validate_graph(workflow)
        assert by_type["UNETLoader"]["widgets_values"] == ["", "default"]
        assert by_type["CLIPLoader"]["widgets_values"] == ["", "krea2", "default"]
        assert by_type["VAELoader"]["widgets_values"] == [""]
        assert "ConditioningZeroOut" in by_type
        source_type = "VAEDecode" if prefix == "gen" else "LoadImage"
        assert by_type["UltimateFaceFix"]["inputs"][0]["link"] in by_type[source_type]["outputs"][0]["links"]
        if prefix == "gen":
            assert {"EmptyLatentImage", "KSampler", "VAEDecode"} <= by_type.keys()
            assert by_type["KSampler"]["widgets_values"][2:7] == [8, 1.0, "euler", "simple", 1.0]
        else:
            assert by_type["LoadImage"]["widgets_values"][0] == ""
