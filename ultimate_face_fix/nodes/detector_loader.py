from __future__ import annotations

from comfy_api.latest import io

from ..core.detection import YoloFaceDetector
from .common import FaceFixDetectorType, NO_MODEL, model_options, model_path


class LoadFaceFixDetector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFixDetectorLoader",
            display_name="Load Face Fix Detector (YOLO)",
            category="ultimate face fix",
            description="Loads a YOLO face detector once and exposes it through a reusable connection.",
            inputs=[
                io.Combo.Input(
                    "detector_model",
                    options=model_options("face_fix_detectors", NO_MODEL),
                )
            ],
            outputs=[FaceFixDetectorType.Output("face_detector")],
        )

    @classmethod
    def execute(cls, detector_model) -> io.NodeOutput:
        return io.NodeOutput(YoloFaceDetector(model_path("face_fix_detectors", detector_model)))
