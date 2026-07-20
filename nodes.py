from __future__ import annotations

from comfy_api.latest import io
import comfy.samplers
import folder_paths

from .composite import composite_faces
from .detector import YoloFaceDetector, analyze_faces
from .regions import FaceFixRegions
from .sampling import process_face_crops
from .semantic import SegFaceRunner


FaceFixDetectorType = io.Custom("FACE_FIX_DETECTOR")
FaceFixParserType = io.Custom("FACE_FIX_PARSER")
FaceLandmarkerType = io.Custom("FACE_DETECTION_MODEL")

NO_MODEL = "none available"


def _model_options(folder_name: str, empty_value: str) -> list[str]:
    models = folder_paths.get_filename_list(folder_name)
    return models if models else [empty_value]


def _model_path(folder_name: str, model_name: str) -> str:
    if model_name == NO_MODEL:
        raise ValueError(f"No model is installed in ComfyUI's {folder_name} folder")
    return folder_paths.get_full_path_or_raise(folder_name, model_name)


def _analysis_inputs() -> list:
    return [
        io.Image.Input("image"),
        FaceFixDetectorType.Input(
            "face_detector",
            optional=True,
            tooltip="Connect Load Face Fix Detector. It may be omitted only when bboxes is connected.",
        ),
        io.Combo.Input("face_selection", options=["all", "largest", "largest_n"], default="all"),
        io.Combo.Input("detection_quality", options=["maximum", "balanced"], default="maximum"),
        io.Int.Input("max_faces", default=16, min=1, max=64, step=1),
        io.Int.Input("min_face_size", default=24, min=4, max=1024, step=1),
        io.Float.Input("detection_confidence", default=0.25, min=0.01, max=1.0, step=0.01),
        io.Float.Input("context_scale", default=1.8, min=1.1, max=3.0, step=0.05),
        io.Combo.Input("target_resolution", options=["none", "512", "768", "1024", "1536"], default="1024"),
        FaceLandmarkerType.Input(
            "face_landmarker",
            display_name="face_landmarker (optional)",
            optional=True,
            tooltip="Connect ComfyUI's Load Face Detection Model (MediaPipe) output for landmark-aware crops.",
        ),
        io.BoundingBox.Input(
            "bboxes",
            display_name="bboxes (optional override)",
            force_input=True,
            optional=True,
            tooltip="When connected, these boxes replace YOLO detection.",
        ),
    ]


def _composite_inputs() -> list:
    return [
        FaceFixParserType.Input(
            "face_parser",
            optional=True,
            tooltip="Connect Load Face Fix Parser. When omitted, SAM/landmark/geometry fallback masks are used.",
        ),
        io.Combo.Input("mask_preset", options=["core_face", "portrait_safe", "head"], default="core_face"),
        io.Combo.Input("sam_refine", options=["auto", "always", "off"], default="auto"),
        io.Float.Input("mask_grow_percent", default=1.5, min=0.0, max=15.0, step=0.1),
        io.Float.Input("mask_feather_percent", default=2.5, min=0.0, max=15.0, step=0.1),
        io.Float.Input("color_match_strength", default=0.65, min=0.0, max=1.0, step=0.05),
        io.Combo.Input("blend_mode", options=["multiband", "alpha"], default="multiband"),
        io.Model.Input(
            "sam_model",
            display_name="SAM 3/3.1 model (optional)",
            optional=True,
            tooltip="Connect the MODEL output from the supplied SAM 3.1 checkpoint.",
        ),
    ]


def _analyze(
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
) -> tuple[FaceFixRegions, object]:
    regions, crops = analyze_faces(
        images=image,
        face_detector=face_detector,
        face_landmarker=face_landmarker,
        external_bboxes=bboxes,
        selection=face_selection,
        max_faces=max_faces,
        min_face_size=min_face_size,
        confidence=detection_confidence,
        context_scale=context_scale,
        target_size=None if target_resolution == "none" else int(target_resolution),
        quality=detection_quality,
    )
    return regions, crops


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
                    options=_model_options("face_fix_detectors", NO_MODEL),
                )
            ],
            outputs=[FaceFixDetectorType.Output("face_detector")],
        )

    @classmethod
    def execute(cls, detector_model) -> io.NodeOutput:
        return io.NodeOutput(YoloFaceDetector(_model_path("face_fix_detectors", detector_model)))


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
                    options=_model_options("face_fix_parsers", NO_MODEL),
                )
            ],
            outputs=[FaceFixParserType.Output("face_parser")],
        )

    @classmethod
    def execute(cls, parser_model) -> io.NodeOutput:
        return io.NodeOutput(SegFaceRunner(_model_path("face_fix_parsers", parser_model)))


class UltimateFaceFix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UltimateFaceFix",
            display_name="Ultimate Face Fix",
            category="ultimate face fix",
            description="Detects, reconstructs, semantically masks, and seamlessly blends one or many faces.",
            inputs=[
                io.Image.Input("image"),
                io.Model.Input("model", tooltip="Generation model used for crop img2img."),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                *_analysis_inputs()[1:],
                io.Combo.Input("repair_mode", options=["detail", "repair", "reconstruct", "custom"], default="repair"),
                io.Float.Input("custom_denoise", default=0.42, min=0.0, max=1.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=25, min=1, max=200, step=1),
                io.Float.Input("cfg", default=5.5, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="beta"),
                *_composite_inputs(),
            ],
            outputs=[
                io.Image.Output("fixed_image"),
                io.Image.Output("original_face_crops"),
                io.Image.Output("processed_face_crops"),
                io.Mask.Output("face_mask"),
                io.Image.Output("debug_preview"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        model,
        vae,
        positive,
        negative,
        face_selection,
        detection_quality,
        max_faces,
        min_face_size,
        detection_confidence,
        context_scale,
        target_resolution,
        repair_mode,
        custom_denoise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        mask_preset,
        sam_refine,
        mask_grow_percent,
        mask_feather_percent,
        color_match_strength,
        blend_mode,
        face_detector=None,
        face_parser=None,
        face_landmarker=None,
        bboxes=None,
        sam_model=None,
    ) -> io.NodeOutput:
        regions, crops = _analyze(
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
        processed = process_face_crops(
            crops,
            regions,
            model,
            vae,
            positive,
            negative,
            repair_mode,
            custom_denoise,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
        )
        fixed, mask, preview = composite_faces(
            image,
            regions,
            processed,
            face_parser,
            sam_model,
            mask_preset,
            sam_refine,
            mask_grow_percent,
            mask_feather_percent,
            color_match_strength,
            blend_mode,
        )
        return io.NodeOutput(fixed, crops, processed, mask, preview)
