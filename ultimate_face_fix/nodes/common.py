from __future__ import annotations

from comfy_api.latest import io
import comfy.samplers
import folder_paths


FaceFixDetectorType = io.Custom("FACE_FIX_DETECTOR")
FaceFixParserType = io.Custom("FACE_FIX_PARSER")
FaceFixContextType = io.Custom("ULTIMATE_FACE_FIX_CONTEXT")
FaceLandmarkerType = io.Custom("FACE_DETECTION_MODEL")

NO_MODEL = "none available"


def model_options(folder_name: str, empty_value: str) -> list[str]:
    models = folder_paths.get_filename_list(folder_name)
    return models if models else [empty_value]


def model_path(folder_name: str, model_name: str) -> str:
    if model_name == NO_MODEL:
        raise ValueError(f"No model is installed in ComfyUI's {folder_name} folder")
    return folder_paths.get_full_path_or_raise(folder_name, model_name)


def analysis_inputs() -> list:
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


def composite_inputs() -> list:
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


def sampling_inputs() -> list:
    return [
        io.Combo.Input("repair_mode", options=["detail", "repair", "reconstruct", "custom"], default="custom"),
        io.Float.Input("custom_denoise", default=0.40, min=0.0, max=1.0, step=0.01),
        io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
        io.Int.Input("steps", default=25, min=1, max=200, step=1),
        io.Float.Input("cfg", default=5.5, min=0.0, max=100.0, step=0.1),
        io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
        io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="beta"),
    ]
