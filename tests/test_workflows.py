from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


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
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
    return {node["type"]: node for node in workflow["nodes"]}


def test_single_face_ui_workflow_is_complete():
    by_type = _validate_graph(_load_workflow("single_face.json"))
    assert {
        "LoadImage",
        "CheckpointLoaderSimple",
        "UltimateFaceFixDetectorLoader",
        "UltimateFaceFixParserLoader",
        "LoadMediaPipeFaceLandmarker",
        "UltimateFaceFix",
        "SaveImage",
    } <= by_type.keys()
    settings = by_type["UltimateFaceFix"]["widgets_values"]
    assert settings[0] == "largest"
    assert settings[2] == 1
    assert settings[6] == "1024"
    assert settings[13:15] == ["euler", "beta"]


def test_multi_face_ui_workflow_uses_practical_defaults():
    by_type = _validate_graph(_load_workflow("multi_face.json"))
    settings = by_type["UltimateFaceFix"]["widgets_values"]
    assert settings[0] == "all"
    assert settings[2] == 8
    assert settings[6] == "768"
    assert settings[13:15] == ["euler", "beta"]
    assert by_type["UltimateFaceFixDetectorLoader"]["widgets_values"] == ["face_yolov9c.pt"]
    assert by_type["UltimateFaceFixParserLoader"]["widgets_values"] == [
        "segface_convnext_celeba_512.safetensors"
    ]
