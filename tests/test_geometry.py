from __future__ import annotations

import torch

from ultimate_face_fix.geometry import (
    box_iou,
    convex_hull,
    extract_region_crop,
    extract_square,
    make_region,
    nms,
    reflection_indices,
    square_bounds,
)


def test_reflection_indices_support_padding_larger_than_image():
    indices = reflection_indices(-7, 18, 3, torch.device("cpu"))
    assert indices.min() >= 0
    assert indices.max() < 3
    assert indices.tolist()[:6] == [1, 2, 1, 0, 1, 2]


def test_extract_square_reflects_border_and_keeps_shape():
    image = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4, 1)
    crop = extract_square(image, (-2, -2, 4, 4))
    assert crop.shape == (6, 6, 1)
    assert crop[2, 2].item() == image[0, 0].item()


def test_square_bounds_context_and_upward_bias():
    bounds = square_bounds((40.0, 50.0, 60.0, 80.0), 2.0)
    assert bounds[2] - bounds[0] == 60
    assert bounds[3] - bounds[1] == 60
    assert (bounds[1] + bounds[3]) / 2 < 65


def test_region_transforms_round_trip_and_crop_size():
    region = make_region(
        0,
        0,
        0.9,
        (10.0, 20.0, 30.0, 44.0),
        (11.0, 19.0, 31.0, 45.0),
        ((12.0, 21.0), (30.0, 21.0), (21.0, 44.0)),
        1.8,
        128,
        64,
        64,
    )
    x, y = 21.0, 32.0
    forward = region.forward_transform
    inverse = region.inverse_transform
    crop_x = forward[0][0] * x + forward[0][2]
    crop_y = forward[1][1] * y + forward[1][2]
    assert abs(inverse[0][0] * crop_x + inverse[0][2] - x) < 1e-6
    assert abs(inverse[1][1] * crop_y + inverse[1][2] - y) < 1e-6
    image = torch.rand(64, 64, 3)
    assert extract_region_crop(image, region).shape == (128, 128, 3)


def test_nms_and_iou_are_deterministic():
    candidates = [
        ((0.0, 0.0, 20.0, 20.0), 0.8),
        ((1.0, 1.0, 21.0, 21.0), 0.9),
        ((30.0, 30.0, 40.0, 40.0), 0.7),
    ]
    kept = nms(candidates, 0.45)
    assert len(kept) == 2
    assert kept[0][1] == 0.9
    assert box_iou(candidates[0][0], candidates[1][0]) > 0.8


def test_convex_hull_removes_interior_points():
    hull = convex_hull(((0, 0), (2, 0), (2, 2), (0, 2), (1, 1)))
    assert set(hull) == {(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)}
