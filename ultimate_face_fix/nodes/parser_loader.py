from __future__ import annotations

from comfy_api.latest import io

from ..core.segmentation import SegFaceRunner
from .common import FaceFixParserType, NO_MODEL, model_options, model_path


class LoadFaceFixParser(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFixParserLoader",
            display_name="Load Face Fix Parser (SegFace)",
            category="ultimate face fix",
            description="Loads and patches the SegFace parser once for reuse by face-fix nodes.",
            inputs=[
                io.Combo.Input(
                    "parser_model",
                    options=model_options("face_fix_parsers", NO_MODEL),
                )
            ],
            outputs=[FaceFixParserType.Output("face_parser")],
        )

    @classmethod
    def execute(cls, parser_model) -> io.NodeOutput:
        return io.NodeOutput(SegFaceRunner(model_path("face_fix_parsers", parser_model)))
