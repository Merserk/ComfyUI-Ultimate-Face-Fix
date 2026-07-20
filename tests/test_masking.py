from __future__ import annotations

import torch

from ultimate_face_fix.geometry import make_region
from ultimate_face_fix.masking import build_face_mask, dilate, erode, geometry_mask
from ultimate_face_fix.semantic import semantic_mask


def _region(with_landmarks=True):
    points = ((30.0, 25.0), (70.0, 25.0), (78.0, 60.0), (50.0, 82.0), (22.0, 60.0)) if with_landmarks else ()
    return make_region(0, 0, 0.9, (20, 20, 80, 85), (20, 20, 80, 85), points, 1.5, 128, 100, 100)


def test_mask_presets_have_expected_coverage():
    region = _region(False)
    core = geometry_mask(region, "core_face")
    portrait = geometry_mask(region, "portrait_safe")
    head = geometry_mask(region, "head")
    assert core.sum() < portrait.sum() < head.sum()


def test_landmark_fallback_builds_soft_nonempty_mask():
    region = _region(True)
    original = torch.rand(128, 128, 3)
    processed = torch.rand(128, 128, 3)
    alpha, report = build_face_mask(
        original,
        processed,
        region,
        None,
        None,
        "core_face",
        "off",
        1.5,
        2.5,
    )
    assert report["source"] == "landmarks"
    assert alpha.shape == (128, 128)
    assert 0 < alpha.sum() < 128 * 128
    assert alpha.max() <= 1 and alpha.min() >= 0


def test_semantic_mask_class_presets_select_expected_features():
    labels = torch.tensor([[[2, 15], [14, 0]]])  # skin, glasses, hair, background
    core = semantic_mask(labels, "core_face", 2)
    portrait = semantic_mask(labels, "portrait_safe", 2)
    head = semantic_mask(labels, "head", 2)
    assert core.sum() == 1
    assert portrait.sum() == 2
    assert head.sum() == 3


def test_morphology_growth_and_erosion_are_scale_bounded():
    mask = torch.zeros(11, 11)
    mask[5, 5] = 1
    grown = dilate(mask, 2)
    assert grown.sum() == 25
    assert torch.equal(erode(grown, 2), mask)
