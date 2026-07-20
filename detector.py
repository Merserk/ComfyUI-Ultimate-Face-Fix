from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import (
    Box,
    box_area,
    box_iou,
    clip_box,
    extract_region_crop,
    extract_square,
    make_region,
    nms,
    stack_square_crops,
    square_bounds,
)
from .regions import FaceFixRegions


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


class YoloFaceDetector:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Face detector was not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultimate Face Fix requires ultralytics. Run the bundled install command from README.md."
            ) from exc
        self.model_path = str(path)
        self.model_name = path.name
        self.model = YOLO(self.model_path, task="detect")


def _uint8_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = image[..., :3].mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(rgb[..., ::-1])


def _tile_origins(length: int, tile_size: int, overlap: float = 0.25) -> list[int]:
    if tile_size >= length:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _results_to_detections(
    result,
    offset_x: int = 0,
    offset_y: int = 0,
    tile_size: tuple[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[Detection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    coords = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    detections = []
    for (x1, y1, x2, y2), score in zip(coords, scores):
        if tile_size is not None and image_size is not None:
            tile_width, tile_height = tile_size
            image_width, image_height = image_size
            margin = max(3.0, min(tile_width, tile_height) * 0.008)
            touches_internal_edge = (
                (x1 <= margin and offset_x > 0)
                or (y1 <= margin and offset_y > 0)
                or (x2 >= tile_width - margin and offset_x + tile_width < image_width)
                or (y2 >= tile_height - margin and offset_y + tile_height < image_height)
            )
            if touches_internal_edge:
                continue
        detections.append(
            Detection(
                (float(x1 + offset_x), float(y1 + offset_y), float(x2 + offset_x), float(y2 + offset_y)),
                float(score),
            )
        )
    return detections


def detect_yolo_faces(
    image: torch.Tensor,
    detector: YoloFaceDetector,
    confidence: float,
    min_face_size: int,
    max_faces: int,
    quality: str,
) -> list[Detection]:
    source = _uint8_bgr(image)
    height, width = source.shape[:2]
    model = detector.model
    full = model.predict(
        source=source,
        imgsz=1280 if quality == "maximum" else 960,
        conf=confidence,
        iou=0.45,
        max_det=max(64, max_faces * 4),
        device="cpu",
        verbose=False,
    )[0]
    detections = _results_to_detections(full)

    if quality == "maximum" and max(height, width) >= 640:
        tile_size = min(max(height, width), max(512, int(round(max(height, width) * 0.65))))
        tile_h = min(tile_size, height)
        tile_w = min(tile_size, width)
        tiles: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for top in _tile_origins(height, tile_h):
            for left in _tile_origins(width, tile_w):
                tiles.append(np.ascontiguousarray(source[top : top + tile_h, left : left + tile_w]))
                offsets.append((left, top))
        if len(tiles) > 1:
            tile_results = model.predict(
                source=tiles,
                imgsz=960,
                conf=max(0.05, confidence * 0.85),
                iou=0.45,
                max_det=max(32, max_faces * 3),
                device="cpu",
                verbose=False,
            )
            for result, (left, top) in zip(tile_results, offsets):
                detections.extend(
                    _results_to_detections(
                        result,
                        left,
                        top,
                        tile_size=(tile_w, tile_h),
                        image_size=(width, height),
                    )
                )

    candidates = []
    for detection in detections:
        clipped = clip_box(detection.box, width, height)
        if min(clipped[2] - clipped[0], clipped[3] - clipped[1]) >= min_face_size:
            candidates.append((clipped, detection.confidence))
    return [Detection(box, score) for box, score in nms(candidates, 0.45)]


def normalize_external_boxes(bboxes, batch_size: int) -> list[list[Detection]]:
    if bboxes is None:
        return [[] for _ in range(batch_size)]
    if isinstance(bboxes, dict):
        frames = [[bboxes] for _ in range(batch_size)]
    elif isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list):
        frames = list(bboxes)
        while len(frames) < batch_size:
            frames.append([])
    elif isinstance(bboxes, list):
        frames = [bboxes for _ in range(batch_size)]
    else:
        raise ValueError("BOUNDING_BOX input must be a box, a list of boxes, or a per-image list of boxes")

    normalized: list[list[Detection]] = []
    for frame in frames[:batch_size]:
        per_frame = []
        for item in frame:
            x = float(item["x"])
            y = float(item["y"])
            per_frame.append(
                Detection(
                    (x, y, x + float(item["width"]), y + float(item["height"])),
                    float(item.get("score", 1.0)),
                )
            )
        normalized.append(per_frame)
    return normalized


def _landmark_refinement(image: torch.Tensor, detection: Detection, face_landmarker):
    if face_landmarker is None:
        return detection.box, ()
    provisional_bounds = square_bounds(detection.box, 1.35, upward_bias=0.03)
    provisional = extract_square(image, provisional_bounds)
    rgb = provisional[..., :3].mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()

    result = face_landmarker.detect_batch([rgb], num_faces=1, score_thresh=0.35, variant="short")
    if not result or not result[0]:
        result = face_landmarker.detect_batch([rgb], num_faces=1, score_thresh=0.30, variant="full")
    if not result or not result[0]:
        return detection.box, ()

    face = result[0][0]
    left, top, _, _ = provisional_bounds
    local_box = face["bbox_xyxy"]
    mapped_box = (
        float(local_box[0] + left),
        float(local_box[1] + top),
        float(local_box[2] + left),
        float(local_box[3] + top),
    )
    mapped_box = clip_box(mapped_box, image.shape[1], image.shape[0])
    if box_area(mapped_box) <= 0.0 or box_iou(mapped_box, detection.box) < 0.05:
        return detection.box, ()

    landmarks = tuple((float(point[0] + left), float(point[1] + top)) for point in face["landmarks_xy"])
    return mapped_box, landmarks


def _spatial_sort(detections: list[Detection], image_height: int) -> list[Detection]:
    row_height = max(1.0, image_height * 0.10)
    return sorted(
        detections,
        key=lambda detection: (
            int(((detection.box[1] + detection.box[3]) * 0.5) // row_height),
            (detection.box[0] + detection.box[2]) * 0.5,
            detection.box[1],
            -detection.confidence,
        ),
    )


def analyze_faces(
    images: torch.Tensor,
    face_detector: YoloFaceDetector | None,
    face_landmarker,
    external_bboxes,
    selection: str,
    max_faces: int,
    min_face_size: int,
    confidence: float,
    context_scale: float,
    target_size: int | None,
    quality: str,
) -> tuple[FaceFixRegions, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"IMAGE must be BHWC, got {tuple(images.shape)}")
    batch_size = images.shape[0]
    supplied = normalize_external_boxes(external_bboxes, batch_size) if external_bboxes is not None else None
    regions = []
    crops = []

    for source_index, image in enumerate(images):
        height, width = image.shape[:2]
        if supplied is not None:
            detections = [
                Detection(clip_box(item.box, width, height), item.confidence)
                for item in supplied[source_index]
                if min(item.box[2] - item.box[0], item.box[3] - item.box[1]) >= min_face_size
            ]
        else:
            if face_detector is None:
                raise ValueError("A YOLO detector is required when no external BOUNDING_BOX input is connected")
            detections = detect_yolo_faces(image, face_detector, confidence, min_face_size, max_faces, quality)

        if selection == "largest":
            detections = sorted(detections, key=lambda item: (-box_area(item.box), -item.confidence))[:1]
        elif selection == "largest_n":
            detections = sorted(detections, key=lambda item: (-box_area(item.box), -item.confidence))[:max_faces]
        else:
            detections = _spatial_sort(detections, height)[:max_faces]
        detections = _spatial_sort(detections, height)

        for local_index, detection in enumerate(detections):
            refined_box, landmarks = _landmark_refinement(image, detection, face_landmarker)
            region = make_region(
                source_index=source_index,
                face_index=len(regions),
                confidence=detection.confidence,
                detection_box=detection.box,
                refined_box=refined_box,
                landmarks=landmarks,
                context_scale=context_scale,
                target_size=target_size,
                image_width=width,
                image_height=height,
            )
            regions.append(region)
            crops.append(extract_region_crop(image, region))

    result = FaceFixRegions(
        regions=tuple(regions),
        image_shapes=tuple((int(image.shape[0]), int(image.shape[1])) for image in images),
        target_size=target_size,
        detector_name="external_bboxes" if supplied is not None else face_detector.model_name,
        selection=selection,
    )
    if crops:
        crop_batch = stack_square_crops(crops)
    else:
        placeholder_size = 1 if target_size is None else target_size
        crop_batch = torch.zeros((1, placeholder_size, placeholder_size, 3), dtype=images.dtype, device=images.device)
    return result, crop_batch
