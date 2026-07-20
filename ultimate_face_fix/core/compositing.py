from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.color import lab_to_rgb, rgb_to_lab

from .geometry import extract_region_crop
from .masking import build_face_mask, dilate, erode
from .types import FaceFixRegions
from .segmentation import SegFaceRunner
from .previews import mask_preview


def _weighted_stats(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mean = (values * weights).sum(dim=(2, 3), keepdim=True) / total
    variance = ((values - mean).square() * weights).sum(dim=(2, 3), keepdim=True) / total
    return mean, variance.sqrt().clamp_min(1e-4)


def local_color_match(
    original: torch.Tensor,
    processed: torch.Tensor,
    alpha: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0:
        return processed
    mask_area = max(1.0, float((alpha > 0.25).sum().item()))
    radius = max(2, min(16, int(round(mask_area ** 0.25))))
    ring = (dilate((alpha > 0.35).float(), radius) - erode((alpha > 0.35).float(), radius)).clamp(0.0, 1.0)
    if ring.sum() < 16:
        ring = (alpha > 0.1).float()
    weights = ring[None, None]
    original_lab = rgb_to_lab(original.movedim(-1, 0).unsqueeze(0).clamp(0.0, 1.0))
    processed_lab = rgb_to_lab(processed.movedim(-1, 0).unsqueeze(0).clamp(0.0, 1.0))
    original_mean, original_std = _weighted_stats(original_lab, weights)
    processed_mean, processed_std = _weighted_stats(processed_lab, weights)
    scale = (original_std / processed_std).clamp(0.75, 1.25)
    adjusted_lab = (processed_lab - processed_mean) * scale + original_mean
    mixed_lab = processed_lab.lerp(adjusted_lab, strength)
    return lab_to_rgb(mixed_lab).clamp(0.0, 1.0)[0].movedim(0, -1)


def _pyramid_down(image: torch.Tensor) -> torch.Tensor:
    height = max(1, image.shape[-2] // 2)
    width = max(1, image.shape[-1] // 2)
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False, antialias=True)


def _pyramid_up(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def multiband_blend(original: torch.Tensor, processed: torch.Tensor, alpha: torch.Tensor, levels: int = 4) -> torch.Tensor:
    original_level = original.movedim(-1, 0).unsqueeze(0)
    processed_level = processed.movedim(-1, 0).unsqueeze(0)
    alpha_level = alpha[None, None]
    original_gaussian = [original_level]
    processed_gaussian = [processed_level]
    alpha_gaussian = [alpha_level]
    for _ in range(max(1, levels) - 1):
        if min(original_gaussian[-1].shape[-2:]) <= 8:
            break
        original_gaussian.append(_pyramid_down(original_gaussian[-1]))
        processed_gaussian.append(_pyramid_down(processed_gaussian[-1]))
        alpha_gaussian.append(_pyramid_down(alpha_gaussian[-1]))

    original_laplacian = []
    processed_laplacian = []
    for index in range(len(original_gaussian) - 1):
        target_size = original_gaussian[index].shape[-2:]
        original_laplacian.append(original_gaussian[index] - _pyramid_up(original_gaussian[index + 1], target_size))
        processed_laplacian.append(processed_gaussian[index] - _pyramid_up(processed_gaussian[index + 1], target_size))
    original_laplacian.append(original_gaussian[-1])
    processed_laplacian.append(processed_gaussian[-1])

    blended_levels = [
        old * (1.0 - mask) + new * mask
        for old, new, mask in zip(original_laplacian, processed_laplacian, alpha_gaussian)
    ]
    result = blended_levels[-1]
    for index in range(len(blended_levels) - 2, -1, -1):
        result = _pyramid_up(result, blended_levels[index].shape[-2:]) + blended_levels[index]
    result = result[0].movedim(0, -1).clamp(0.0, 1.0)
    return torch.where(alpha.unsqueeze(-1) > 1e-7, result, original)


def _project_to_source(
    value: torch.Tensor,
    region,
    source_width: int,
    source_height: int,
    channels: int,
) -> tuple[torch.Tensor, tuple[slice, slice], tuple[slice, slice]]:
    side = region.crop_size
    if channels == 1:
        resized = F.interpolate(value[None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
    else:
        resized = F.interpolate(
            value.movedim(-1, 0).unsqueeze(0),
            size=(side, side),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].movedim(0, -1)
    left, top, right, bottom = region.crop_box
    source_x1 = max(0, -left)
    source_y1 = max(0, -top)
    source_x2 = side - max(0, right - source_width)
    source_y2 = side - max(0, bottom - source_height)
    dest_x1 = max(0, left)
    dest_y1 = max(0, top)
    dest_x2 = dest_x1 + max(0, source_x2 - source_x1)
    dest_y2 = dest_y1 + max(0, source_y2 - source_y1)
    return resized, (slice(source_y1, source_y2), slice(source_x1, source_x2)), (slice(dest_y1, dest_y2), slice(dest_x1, dest_x2))


def composite_faces(
    images: torch.Tensor,
    regions: FaceFixRegions,
    processed_crops: torch.Tensor,
    parser: SegFaceRunner | None,
    sam_model,
    mask_preset: str,
    sam_refine: str,
    grow_percent: float,
    feather_percent: float,
    color_match_strength: float,
    blend_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if regions.face_count == 0:
        empty_mask = torch.zeros(images.shape[:3], dtype=images.dtype, device=images.device)
        return images.clone(), empty_mask, mask_preview(images, empty_mask, regions)
    if processed_crops.ndim != 4 or processed_crops.shape[0] != regions.face_count:
        raise ValueError(
            f"Processed crop count must match detected face count: got {processed_crops.shape[0] if processed_crops.ndim else 0}, expected {regions.face_count}"
        )

    delta_accum = torch.zeros_like(images[..., :3])
    weight_accum = torch.zeros(images.shape[:3], dtype=images.dtype, device=images.device)
    alpha_accum = torch.zeros_like(weight_accum)
    for crop_index, region in enumerate(regions.regions):
        image = images[region.source_index]
        source_height, source_width = image.shape[:2]
        original_crop = extract_region_crop(image, region)
        processed_crop = processed_crops[
            crop_index,
            : region.target_size,
            : region.target_size,
        ].to(device=image.device, dtype=image.dtype)
        if processed_crop.shape[:2] != (region.target_size, region.target_size):
            processed_crop = F.interpolate(
                processed_crop.movedim(-1, 0).unsqueeze(0),
                size=(region.target_size, region.target_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0].movedim(0, -1)
        alpha = build_face_mask(
            original_crop,
            processed_crop,
            region,
            parser,
            sam_model,
            mask_preset,
            sam_refine,
            grow_percent,
            feather_percent,
        )
        processed_crop = local_color_match(original_crop, processed_crop, alpha, color_match_strength)
        if blend_mode == "multiband":
            blended = multiband_blend(original_crop, processed_crop, alpha, levels=4)
        else:
            blended = original_crop * (1.0 - alpha.unsqueeze(-1)) + processed_crop * alpha.unsqueeze(-1)
        delta = blended - original_crop

        delta_side, source_slice, dest_slice = _project_to_source(delta, region, source_width, source_height, 3)
        alpha_side, alpha_source_slice, alpha_dest_slice = _project_to_source(
            alpha, region, source_width, source_height, 1
        )
        priority = 0.5 + 0.5 * region.confidence
        delta_accum[region.source_index, dest_slice[0], dest_slice[1]] += delta_side[source_slice] * priority
        weight_accum[region.source_index, alpha_dest_slice[0], alpha_dest_slice[1]] += (
            alpha_side[alpha_source_slice] * priority
        )
        alpha_accum[region.source_index, alpha_dest_slice[0], alpha_dest_slice[1]] += alpha_side[
            alpha_source_slice
        ]
    effective_alpha = alpha_accum.clamp(0.0, 1.0)
    normalized_delta = delta_accum / weight_accum.clamp_min(1e-7).unsqueeze(-1)
    output = images.clone()
    rgb = images[..., :3] + normalized_delta * effective_alpha.unsqueeze(-1)
    output[..., :3] = torch.where(
        effective_alpha.unsqueeze(-1) > 1e-7,
        rgb.clamp(0.0, 1.0),
        images[..., :3],
    )
    return output, effective_alpha, mask_preview(output, effective_alpha, regions)
