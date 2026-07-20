from __future__ import annotations

from comfy_api.latest import io

from ..core.pipeline import analyze
from ..core.types import FaceFixPipelineContext
from .common import FaceFixContextType, analysis_inputs


class UltimateFaceFixExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFixExtract",
            display_name="Ultimate Face Fix (Extract)",
            category="ultimate face fix",
            description="Detects faces and extracts square crops for an optional custom image-processing pipeline.",
            inputs=analysis_inputs(),
            outputs=[
                io.Image.Output("extract_image"),
                FaceFixContextType.Output("face_fix_context"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        face_selection,
        detection_quality,
        max_faces,
        min_face_size,
        detection_confidence,
        context_scale,
        target_resolution,
        face_detector=None,
        face_landmarker=None,
        bboxes=None,
    ) -> io.NodeOutput:
        regions, crops = analyze(
            image=image,
            face_detector=face_detector,
            face_selection=face_selection,
            detection_quality=detection_quality,
            max_faces=max_faces,
            min_face_size=min_face_size,
            detection_confidence=detection_confidence,
            context_scale=context_scale,
            target_resolution=target_resolution,
            face_landmarker=face_landmarker,
            bboxes=bboxes,
        )
        context = FaceFixPipelineContext(
            source_image=image,
            regions=regions,
            original_crops=crops,
            extract_batch_side=int(crops.shape[1]),
        )
        return io.NodeOutput(crops, context)
