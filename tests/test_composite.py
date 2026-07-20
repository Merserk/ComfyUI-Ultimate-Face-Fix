from __future__ import annotations

import pytest
import torch

from ultimate_face_fix.composite import composite_faces, local_color_match, multiband_blend
from ultimate_face_fix.detector import analyze_faces


def _regions(image, boxes, target=64):
    return analyze_faces(
        image,
        None,
        None,
        [boxes],
        "all",
        16,
        4,
        0.25,
        1.6,
        target,
        "balanced",
    )


def test_composite_preserves_every_pixel_outside_mask():
    image = torch.linspace(0, 1, 96).view(1, 96, 1, 1).expand(1, 96, 96, 3).clone()
    regions, crops = _regions(image, [{"x": 30, "y": 25, "width": 30, "height": 38, "score": 0.9}])
    processed = (1.0 - crops).clamp(0, 1)
    output, mask, _, reports = composite_faces(
        image,
        regions,
        processed,
        None,
        None,
        "core_face",
        "off",
        1.5,
        2.5,
        0.0,
        "multiband",
    )
    outside = mask <= 1e-7
    assert torch.equal(output[outside], image[outside])
    assert torch.count_nonzero(output - image) > 0
    assert reports[0]["source"] == "geometry"


def test_overlap_composite_is_bounded_and_deterministic():
    torch.manual_seed(1)
    image = torch.rand(1, 96, 96, 3)
    boxes = [
        {"x": 25, "y": 25, "width": 35, "height": 40, "score": 0.9},
        {"x": 45, "y": 28, "width": 34, "height": 39, "score": 0.8},
    ]
    regions, crops = _regions(image, boxes)
    processed = torch.flip(crops, dims=(2,))
    args = (image, regions, processed, None, None, "core_face", "off", 1.5, 2.5, 0.0, "alpha")
    first, mask, _, _ = composite_faces(*args)
    second, _, _, _ = composite_faces(*args)
    assert torch.equal(first, second)
    assert first.min() >= 0 and first.max() <= 1
    assert mask.max() <= 1


def test_single_face_confidence_does_not_reduce_blend_strength():
    image = torch.zeros(1, 64, 64, 3)
    box = {"x": 18, "y": 16, "width": 28, "height": 30}
    high_regions, high_crops = _regions(image, [{**box, "score": 1.0}])
    low_regions, low_crops = _regions(image, [{**box, "score": 0.1}])
    high, high_mask, _, _ = composite_faces(
        image, high_regions, torch.ones_like(high_crops), None, None, "core_face", "off", 0, 0, 0, "alpha"
    )
    low, low_mask, _, _ = composite_faces(
        image, low_regions, torch.ones_like(low_crops), None, None, "core_face", "off", 0, 0, 0, "alpha"
    )
    assert torch.equal(high_mask, low_mask)
    assert torch.allclose(high, low, atol=1e-6)


def test_crop_count_mismatch_is_actionable():
    image = torch.rand(1, 64, 64, 3)
    regions, _ = _regions(image, [{"x": 10, "y": 10, "width": 20, "height": 20}])
    with pytest.raises(ValueError, match="Processed crop count"):
        composite_faces(image, regions, torch.zeros(2, 64, 64, 3), None, None, "core_face", "off", 0, 0, 0, "alpha")


def test_multiband_blend_shape_and_limits():
    original = torch.zeros(64, 64, 3)
    processed = torch.ones(64, 64, 3)
    alpha = torch.zeros(64, 64)
    alpha[16:48, 16:48] = 1
    result = multiband_blend(original, processed, alpha)
    assert result.shape == original.shape
    assert torch.equal(result[alpha == 0], original[alpha == 0])
    assert result.min() >= 0 and result.max() <= 1


def test_local_lab_color_match_reduces_boundary_color_error():
    x = torch.linspace(0.25, 0.55, 64).view(1, 64, 1).expand(64, 64, 3)
    original = x.clone()
    processed = (x + torch.tensor([0.18, -0.08, 0.12])).clamp(0, 1)
    alpha = torch.zeros(64, 64)
    alpha[12:52, 12:52] = 1
    matched = local_color_match(original, processed, alpha, 1.0)
    ring = (alpha > 0.5).unsqueeze(-1).expand_as(original)
    before_error = (processed - original).abs()[ring].mean()
    after_error = (matched - original).abs()[ring].mean()
    assert after_error < before_error
