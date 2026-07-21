from __future__ import annotations

import torch

from .detection import YoloFaceDetector, _spatial_sort, detect_yolo_faces
from .geometry import box_area


MAX_FACES = 8
DETECTION_CONFIDENCE = 0.25
MIN_FACE_SIZE = 24
DETECTION_QUALITY = "maximum"


def denoise_for_face_area(area_fraction: float) -> float:
    scaled_area = max(0.0, min(1.0, (area_fraction - 0.006) / 0.014))
    return round(0.40 - scaled_area * 0.22, 3)


def calculate_auto_denoise(
    images: torch.Tensor,
    face_detector: YoloFaceDetector,
) -> tuple[float, int]:
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"IMAGE must be a torch.Tensor, got {type(images).__name__}")
    if images.ndim != 4:
        raise ValueError(f"IMAGE must be BHWC, got shape {tuple(images.shape)}")
    if images.shape[-1] < 3:
        raise ValueError(
            "IMAGE must contain channel-last RGB or RGBA data; "
            f"received shape {tuple(images.shape)}"
        )

    face_denoise_values: list[float] = []
    for image in images:
        height, width = image.shape[:2]
        image_area = float(height * width)
        if image_area <= 0.0:
            raise ValueError(f"IMAGE dimensions must be positive, got {width}x{height}")

        detections = detect_yolo_faces(
            image=image,
            detector=face_detector,
            confidence=DETECTION_CONFIDENCE,
            min_face_size=MIN_FACE_SIZE,
            max_faces=MAX_FACES,
            quality=DETECTION_QUALITY,
        )
        detections = _spatial_sort(detections, height)[:MAX_FACES]
        for detection in detections:
            area_fraction = box_area(detection.box) / image_area
            face_denoise_values.append(denoise_for_face_area(area_fraction))

    if not face_denoise_values:
        return 0.0, 0

    average = round(sum(face_denoise_values) / len(face_denoise_values), 3)
    return average, len(face_denoise_values)
