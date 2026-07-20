from __future__ import annotations

import asyncio

import ultimate_face_fix


def test_v3_extension_registers_all_public_nodes():
    extension = asyncio.run(ultimate_face_fix.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    ids = {node.define_schema().node_id for node in nodes}
    assert ids == {
        "UltimateFaceFixDetectorLoader",
        "UltimateFaceFixParserLoader",
        "UltimateFaceFix",
        "UltimateFaceFixAnalyze",
        "UltimateFaceFixComposite",
    }


def test_models_are_connection_inputs_and_native_resolution_is_available():
    extension = asyncio.run(ultimate_face_fix.comfy_entrypoint())
    node_types = asyncio.run(extension.get_node_list())
    schemas = {node.define_schema().node_id: node.define_schema() for node in node_types}
    ultimate = schemas["UltimateFaceFix"]
    inputs = {item.id: item for item in ultimate.inputs}
    assert inputs["face_detector"].get_io_type() == "FACE_FIX_DETECTOR"
    assert inputs["face_parser"].get_io_type() == "FACE_FIX_PARSER"
    assert "none" in inputs["target_resolution"].options
    assert inputs["sampler_name"].default == "euler"
    assert inputs["scheduler"].default == "beta"
    assert "detector_model" not in inputs
    assert "parser_model" not in inputs
