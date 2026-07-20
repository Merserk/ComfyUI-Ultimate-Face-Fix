from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .regions import FaceRegion


Box = tuple[float, float, float, float]


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: Box, b: Box) -> float:
    inter = (
        max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    )
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def nms(boxes: Iterable[tuple[Box, float]], threshold: float = 0.45) -> list[tuple[Box, float]]:
    ordered = sorted(boxes, key=lambda item: (-item[1], item[0][1], item[0][0]))
    kept: list[tuple[Box, float]] = []
    for candidate in ordered:
        if all(box_iou(candidate[0], existing[0]) <= threshold for existing in kept):
            kept.append(candidate)
    return kept


def square_bounds(box: Box, context_scale: float, upward_bias: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    side = max(2, int(math.ceil(max(width, height) * context_scale)))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5 - height * upward_bias
    left = int(math.floor(cx - side * 0.5))
    top = int(math.floor(cy - side * 0.5))
    return left, top, left + side, top + side


def reflection_indices(start: int, length: int, limit: int, device: torch.device) -> torch.Tensor:
    if limit <= 1:
        return torch.zeros(length, dtype=torch.long, device=device)
    values = torch.arange(start, start + length, dtype=torch.long, device=device)
    period = 2 * limit - 2
    values = torch.remainder(values, period)
    return torch.where(values < limit, values, period - values)


def extract_square(image: torch.Tensor, bounds: tuple[int, int, int, int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected one HWC image, got shape {tuple(image.shape)}")
    left, top, right, bottom = bounds
    side_x = right - left
    side_y = bottom - top
    if side_x <= 0 or side_x != side_y:
        raise ValueError(f"Crop bounds must describe a positive square, got {bounds}")
    y_indices = reflection_indices(top, side_y, image.shape[0], image.device)
    x_indices = reflection_indices(left, side_x, image.shape[1], image.device)
    return image.index_select(0, y_indices).index_select(1, x_indices)


def resize_image(image: torch.Tensor, width: int, height: int | None = None) -> torch.Tensor:
    height = width if height is None else height
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    channels_first = batched.movedim(-1, 1)
    resized = F.interpolate(channels_first, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    result = resized.movedim(1, -1).clamp(0.0, 1.0)
    return result[0] if image.ndim == 3 else result


def map_box_to_crop(box: Box, bounds: tuple[int, int, int, int], target_size: int) -> Box:
    left, top, right, _ = bounds
    scale = target_size / (right - left)
    return tuple((value - (left if index % 2 == 0 else top)) * scale for index, value in enumerate(box))  # type: ignore[return-value]


def map_points_to_crop(points: Iterable[tuple[float, float]], bounds: tuple[int, int, int, int], target_size: int) -> tuple[tuple[float, float], ...]:
    left, top, right, _ = bounds
    scale = target_size / (right - left)
    return tuple(((x - left) * scale, (y - top) * scale) for x, y in points)


def convex_hull(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) <= 2:
        return tuple(unique)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def make_region(
    source_index: int,
    face_index: int,
    confidence: float,
    detection_box: Box,
    refined_box: Box,
    landmarks: Iterable[tuple[float, float]],
    context_scale: float,
    target_size: int | None,
    image_width: int,
    image_height: int,
) -> FaceRegion:
    bounds = square_bounds(refined_box, context_scale)
    left, top, right, bottom = bounds
    side = right - left
    processing_size = side if target_size is None else target_size
    scale = processing_size / side
    forward = (
        (scale, 0.0, -left * scale),
        (0.0, scale, -top * scale),
        (0.0, 0.0, 1.0),
    )
    inverse = (
        (1.0 / scale, 0.0, float(left)),
        (0.0, 1.0 / scale, float(top)),
        (0.0, 0.0, 1.0),
    )
    padding = (
        max(0, -left),
        max(0, -top),
        max(0, right - image_width),
        max(0, bottom - image_height),
    )
    landmark_points = tuple((float(x), float(y)) for x, y in landmarks)
    hull = convex_hull(landmark_points)
    seed_offset = (
        (source_index + 1) * 1_000_003
        + int(round(detection_box[0])) * 9_176
        + int(round(detection_box[1])) * 6_113
        + face_index * 97
    ) & 0xFFFFFFFFFFFFFFFF
    return FaceRegion(
        source_index=source_index,
        face_index=face_index,
        confidence=float(confidence),
        detection_box=detection_box,
        refined_box=refined_box,
        crop_box=bounds,
        crop_size=side,
        target_size=processing_size,
        padding=padding,
        forward_transform=forward,
        inverse_transform=inverse,
        face_box_crop=map_box_to_crop(refined_box, bounds, processing_size),
        landmarks=landmark_points,
        landmark_hull_crop=map_points_to_crop(hull, bounds, processing_size),
        seed_offset=seed_offset,
    )


def extract_region_crop(image: torch.Tensor, region: FaceRegion) -> torch.Tensor:
    return resize_image(extract_square(image, region.crop_box), region.target_size)


def stack_square_crops(crops: list[torch.Tensor]) -> torch.Tensor:
    if not crops:
        raise ValueError("At least one crop is required")
    batch_side = max(crop.shape[0] for crop in crops)
    padded = []
    for crop in crops:
        if crop.ndim != 3 or crop.shape[0] != crop.shape[1]:
            raise ValueError(f"Face crops must be square HWC tensors, got {tuple(crop.shape)}")
        if crop.shape[0] == batch_side:
            padded.append(crop)
            continue
        y_indices = reflection_indices(0, batch_side, crop.shape[0], crop.device)
        x_indices = reflection_indices(0, batch_side, crop.shape[1], crop.device)
        padded.append(crop.index_select(0, y_indices).index_select(1, x_indices))
    return torch.stack(padded)
