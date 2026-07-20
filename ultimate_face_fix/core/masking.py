from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from kornia.filters import gaussian_blur2d

from .geometry import Box, box_area
from .types import FaceRegion
from .segmentation import SegFaceRunner, sam_box_mask, semantic_mask, semantic_mask_valid


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask[None, None], kernel, stride=1, padding=radius)[0, 0]


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return -dilate(-mask, radius)


def _pil_mask(size: int, draw_callback) -> torch.Tensor:
    canvas = Image.new("L", (size, size), 0)
    draw_callback(ImageDraw.Draw(canvas))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).div_(255.0)


def geometry_mask(region: FaceRegion, preset: str) -> torch.Tensor:
    size = region.target_size
    if len(region.landmark_hull_crop) >= 3 and preset != "head":
        points = [(float(x), float(y)) for x, y in region.landmark_hull_crop]
        return _pil_mask(size, lambda draw: draw.polygon(points, fill=255))

    x1, y1, x2, y2 = region.face_box_crop
    width = x2 - x1
    height = y2 - y1
    if preset == "core_face":
        bounds = (x1 + width * 0.04, y1 + height * 0.02, x2 - width * 0.04, y2 - height * 0.01)
    elif preset == "portrait_safe":
        bounds = (x1 - width * 0.08, y1 - height * 0.05, x2 + width * 0.08, y2 + height * 0.04)
    else:
        bounds = (x1 - width * 0.18, y1 - height * 0.38, x2 + width * 0.18, y2 + height * 0.12)
    return _pil_mask(size, lambda draw: draw.ellipse(bounds, fill=255))


def _mask_valid(mask: torch.Tensor | None, face_box: Box, min_ratio: float, max_ratio: float) -> bool:
    if mask is None:
        return False
    area = float(mask.sum().item())
    ratio = area / max(1.0, box_area(face_box))
    height, width = mask.shape[-2:]
    center_x = max(0, min(width - 1, int(round((face_box[0] + face_box[2]) * 0.5))))
    center_y = max(0, min(height - 1, int(round((face_box[1] + face_box[3]) * 0.5))))
    center_hit = bool(mask[center_y, center_x] > 0.5)
    return min_ratio <= ratio <= max_ratio and (center_hit or ratio >= 0.45)


def _keep_primary_component(mask: torch.Tensor, face_box: Box) -> torch.Tensor:
    binary = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return mask
    center_x = max(0, min(binary.shape[1] - 1, int(round((face_box[0] + face_box[2]) * 0.5))))
    center_y = max(0, min(binary.shape[0] - 1, int(round((face_box[1] + face_box[3]) * 0.5))))
    center_label = int(labels[center_y, center_x])
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    selected = center_label if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] >= areas.max() * 0.1 else largest
    result = torch.from_numpy((labels == selected).astype(np.float32)).to(mask.device)
    return result


def _soften(mask: torch.Tensor, face_width: float, grow_percent: float, feather_percent: float) -> torch.Tensor:
    grow_px = max(0, int(round(face_width * grow_percent / 100.0)))
    hard = dilate(mask, grow_px).clamp(0.0, 1.0)
    if feather_percent <= 0.0:
        return hard
    sigma = max(0.5, face_width * feather_percent / 100.0)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    max_radius = max(1, (min(mask.shape) - 1) // 2)
    radius = min(radius, max_radius)
    kernel = radius * 2 + 1
    softened = gaussian_blur2d(hard[None, None], (kernel, kernel), (sigma, sigma))[0, 0]
    return softened.clamp(0.0, 1.0)


def build_face_mask(
    original_crop: torch.Tensor,
    processed_crop: torch.Tensor,
    region: FaceRegion,
    parser: SegFaceRunner | None,
    sam_model,
    preset: str,
    sam_refine: str,
    grow_percent: float,
    feather_percent: float,
) -> torch.Tensor:
    size = region.target_size
    face_box = region.face_box_crop
    geometric = geometry_mask(region, preset)
    device = original_crop.device
    geometric = geometric.to(device)

    semantic_original = semantic_processed = None
    semantic_original_valid = semantic_processed_valid = False
    if parser is not None:
        labels, confidence = parser.parse(torch.stack([original_crop, processed_crop]))
        masks = semantic_mask(labels, preset, size).to(device)
        semantic_original, semantic_processed = masks[0], masks[1]
        semantic_original_valid = semantic_mask_valid(
            semantic_original.unsqueeze(0), confidence[0:1], face_box
        )
        semantic_processed_valid = semantic_mask_valid(
            semantic_processed.unsqueeze(0), confidence[1:2], face_box
        )

    sam_original = sam_processed = None
    sam_original_valid = sam_processed_valid = False
    run_sam = sam_model is not None and (
        sam_refine == "always" or (sam_refine == "auto" and not (semantic_original_valid and semantic_processed_valid))
    )
    if run_sam:
        sam_original = sam_box_mask(original_crop, face_box, sam_model)
        sam_processed = sam_box_mask(processed_crop, face_box, sam_model)
        if sam_original is not None:
            sam_original = sam_original.to(device)
        if sam_processed is not None:
            sam_processed = sam_processed.to(device)
        sam_original_valid = _mask_valid(sam_original, face_box, 0.15, 2.5)
        sam_processed_valid = _mask_valid(sam_processed, face_box, 0.15, 2.5)

    semantic_fused = None
    if semantic_original_valid:
        semantic_fused = semantic_original
        if semantic_processed_valid:
            expansion = max(2, int(round((face_box[2] - face_box[0]) * 0.04)))
            semantic_fused = torch.maximum(
                semantic_original,
                semantic_processed * dilate(semantic_original, expansion),
            )
    elif semantic_processed_valid:
        semantic_fused = semantic_processed

    sam_fused = None
    if sam_original_valid:
        sam_fused = sam_original
    if sam_processed_valid:
        if sam_fused is None:
            sam_fused = sam_processed
        else:
            expansion = max(2, int(round((face_box[2] - face_box[0]) * 0.04)))
            sam_fused = torch.maximum(sam_original, sam_processed * dilate(sam_original, expansion))

    if semantic_fused is not None:
        base = semantic_fused
        if sam_fused is not None:
            constrained = base * dilate(sam_fused, 2)
            if constrained.sum() >= base.sum() * 0.55:
                base = constrained
        if preset != "head":
            geometry_radius = max(2, int(round((face_box[2] - face_box[0]) * 0.05)))
            constrained = base * dilate(geometric, geometry_radius)
            if constrained.sum() >= base.sum() * 0.65:
                base = constrained
    elif sam_fused is not None:
        base = sam_fused * dilate(geometric, max(2, int(round((face_box[2] - face_box[0]) * 0.05))))
    else:
        base = geometric

    base = _keep_primary_component((base > 0.5).float(), face_box)
    face_width = max(1.0, face_box[2] - face_box[0])
    alpha = _soften(base, face_width, grow_percent, feather_percent)
    return alpha
