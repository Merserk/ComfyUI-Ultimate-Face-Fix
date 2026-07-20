from __future__ import annotations

import torch

from .compositing import composite_faces
from .detection import analyze_faces
from .geometry import rescale_regions_for_crop_batch
from .sampling import process_face_crops
from .types import FaceFixPipelineContext, FaceFixRegions


def analyze(
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


def prepare_external_crops(
    extract_image,
    face_fix_context,
) -> tuple[torch.Tensor, FaceFixRegions]:
    if not isinstance(face_fix_context, FaceFixPipelineContext):
        raise ValueError(
            "face_fix_context must come directly from Ultimate Face Fix (Extract)"
        )
    if not isinstance(extract_image, torch.Tensor):
        raise ValueError(
            f"extract_image must be a torch.Tensor IMAGE batch, got {type(extract_image).__name__}"
        )
    if extract_image.ndim != 4:
        raise ValueError(f"extract_image must be BHWC, got shape {tuple(extract_image.shape)}")
    batch, height, width, channels = extract_image.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"extract_image dimensions must be positive, got {height}x{width}")
    if height != width:
        raise ValueError(
            "External face processing must preserve square crops; "
            f"received {width}x{height}"
        )
    if channels not in (3, 4):
        raise ValueError(
            "extract_image must contain channel-last RGB or RGBA images; "
            f"received {channels} channels"
        )
    expected_batch = max(1, face_fix_context.regions.face_count)
    if batch != expected_batch:
        raise ValueError(
            "External face processing must preserve face count and order: "
            f"received {batch} crops, expected {expected_batch}"
        )
    regions = rescale_regions_for_crop_batch(
        face_fix_context.regions,
        face_fix_context.extract_batch_side,
        height,
    )
    return extract_image[..., :3].clamp(0.0, 1.0), regions


def repair_and_composite(
    image,
    original_crops,
    input_crops,
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
    mask_preset,
    sam_refine,
    mask_grow_percent,
    mask_feather_percent,
    color_match_strength,
    blend_mode,
    face_parser=None,
    sam_model=None,
) -> tuple[object, object, object, object, object]:
    processed = process_face_crops(
        input_crops,
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
    return fixed, original_crops, processed, mask, preview
