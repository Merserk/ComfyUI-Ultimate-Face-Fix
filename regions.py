from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FaceRegion:
    source_index: int
    face_index: int
    confidence: float
    detection_box: tuple[float, float, float, float]
    refined_box: tuple[float, float, float, float]
    crop_box: tuple[int, int, int, int]
    crop_size: int
    target_size: int
    padding: tuple[int, int, int, int]
    forward_transform: tuple[tuple[float, float, float], ...]
    inverse_transform: tuple[tuple[float, float, float], ...]
    face_box_crop: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    landmark_hull_crop: tuple[tuple[float, float], ...]
    seed_offset: int

    def report_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 6)
        return data


@dataclass(frozen=True)
class FaceFixRegions:
    regions: tuple[FaceRegion, ...]
    image_shapes: tuple[tuple[int, int], ...]
    target_size: int | None
    detector_name: str
    selection: str

    @property
    def face_count(self) -> int:
        return len(self.regions)

    def report_dict(self) -> dict:
        return {
            "face_count": self.face_count,
            "detector": self.detector_name,
            "selection": self.selection,
            "target_size": self.target_size if self.target_size is not None else "native",
            "faces": [region.report_dict() for region in self.regions],
        }
