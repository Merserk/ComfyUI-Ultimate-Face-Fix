from __future__ import annotations

from comfy_api.latest import io, ui

from ..core.auto_denoise import calculate_auto_denoise
from .common import FaceFixDetectorType


class UltimateFaceFixAutoDenoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFixAutoDenoise",
            display_name="Auto Denoise",
            category="ultimate face fix",
            description="Calculates a face-size-aware denoise value from up to eight detected faces per image.",
            inputs=[
                io.Image.Input("image"),
                FaceFixDetectorType.Input(
                    "face_detector",
                    tooltip="Connect Load Face Fix Detector (YOLO).",
                ),
            ],
            outputs=[
                io.Float.Output(
                    "custom_denoise",
                    tooltip="Connect to custom_denoise while Ultimate Face Fix uses Custom repair mode.",
                )
            ],
        )

    @classmethod
    def execute(cls, image, face_detector) -> io.NodeOutput:
        denoise, face_count = calculate_auto_denoise(image, face_detector)
        text = f"Denoise: {denoise:.3f}" if face_count else "No face is detected"
        return io.NodeOutput(denoise, ui=ui.PreviewText(text))
