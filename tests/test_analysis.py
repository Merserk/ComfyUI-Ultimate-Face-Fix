from __future__ import annotations

import torch

from ultimate_face_fix.detector import _results_to_detections, analyze_faces, normalize_external_boxes


def _box(x, y, width, height, score=0.9):
    return {"x": x, "y": y, "width": width, "height": height, "score": score, "label": "face"}


def test_external_boxes_single_and_multiple_faces():
    images = torch.rand(1, 128, 160, 3)
    regions, crops = analyze_faces(
        images,
        None,
        None,
        [[_box(15, 20, 30, 35), _box(90, 22, 28, 34)]],
        "all",
        16,
        8,
        0.25,
        1.8,
        64,
        "balanced",
    )
    assert regions.face_count == 2
    assert crops.shape == (2, 64, 64, 3)
    assert regions.regions[0].detection_box[0] < regions.regions[1].detection_box[0]


def test_largest_selection_and_border_padding():
    images = torch.rand(1, 64, 64, 3)
    regions, _ = analyze_faces(
        images,
        None,
        None,
        [[_box(0, 0, 16, 16), _box(30, 30, 24, 24)]],
        "largest",
        16,
        4,
        0.25,
        2.0,
        64,
        "balanced",
    )
    assert regions.face_count == 1
    assert regions.regions[0].detection_box == (30.0, 30.0, 54.0, 54.0)
    assert regions.regions[0].padding[2] > 0 or regions.regions[0].padding[3] > 0


def test_no_faces_returns_placeholder_crop():
    images = torch.rand(2, 32, 32, 3)
    regions, crops = analyze_faces(
        images,
        None,
        None,
        [[], []],
        "all",
        16,
        4,
        0.25,
        1.8,
        64,
        "balanced",
    )
    assert regions.face_count == 0
    assert crops.shape == (1, 64, 64, 3)
    assert torch.count_nonzero(crops) == 0


def test_external_box_normalization_shapes():
    shared = normalize_external_boxes([_box(1, 2, 3, 4)], 2)
    assert len(shared) == 2 and len(shared[0]) == 1 and len(shared[1]) == 1
    per_frame = normalize_external_boxes([[_box(1, 2, 3, 4)], []], 2)
    assert [len(frame) for frame in per_frame] == [1, 0]


def test_tiled_detection_rejects_internal_edge_truncation_only():
    class Boxes:
        xyxy = torch.tensor([[0, 10, 100, 120], [20, 20, 198, 130], [25, 25, 150, 150]])
        conf = torch.tensor([0.9, 0.8, 0.7])

        def __len__(self):
            return len(self.xyxy)

    class Result:
        boxes = Boxes()

    interior = _results_to_detections(Result(), 100, 100, tile_size=(200, 200), image_size=(500, 500))
    assert len(interior) == 1
    assert interior[0].box == (125.0, 125.0, 250.0, 250.0)

    outer = _results_to_detections(Result(), 0, 0, tile_size=(200, 200), image_size=(500, 500))
    assert len(outer) == 2


def test_tiny_faces_are_excluded_and_region_seeds_are_repeatable():
    image = torch.rand(1, 96, 128, 3)
    boxes = [[_box(70, 18, 22, 24), _box(10, 10, 5, 5), _box(20, 20, 24, 26)]]
    first, _ = analyze_faces(image, None, None, boxes, "all", 16, 8, 0.25, 1.8, 64, "balanced")
    second, _ = analyze_faces(image, None, None, boxes, "all", 16, 8, 0.25, 1.8, 64, "balanced")
    assert first.face_count == 2
    assert [region.detection_box for region in first.regions] == [region.detection_box for region in second.regions]
    assert [region.seed_offset for region in first.regions] == [region.seed_offset for region in second.regions]


def test_native_resolution_preserves_each_crop_size_without_upscaling():
    image = torch.rand(1, 180, 220, 3)
    boxes = [[_box(20, 25, 24, 30), _box(105, 35, 48, 54)]]
    regions, crops = analyze_faces(image, None, None, boxes, "all", 16, 8, 0.25, 1.8, None, "balanced")
    sizes = [region.target_size for region in regions.regions]
    assert sizes == [region.crop_size for region in regions.regions]
    assert sizes[0] != sizes[1]
    assert crops.shape[1:3] == (max(sizes), max(sizes))
    assert regions.target_size is None


def test_fixed_resolution_still_produces_uniform_requested_crops():
    image = torch.rand(1, 180, 220, 3)
    boxes = [[_box(20, 25, 24, 30), _box(105, 35, 48, 54)]]
    regions, crops = analyze_faces(image, None, None, boxes, "all", 16, 8, 0.25, 1.8, 768, "balanced")
    assert [region.target_size for region in regions.regions] == [768, 768]
    assert crops.shape == (2, 768, 768, 3)
