from __future__ import annotations

import nodes
import torch

from ultimate_face_fix.detector import analyze_faces
from ultimate_face_fix.sampling import process_face_crops


def test_native_resolution_sampling_uses_each_unpadded_crop(monkeypatch):
    image = torch.rand(1, 180, 220, 3)
    boxes = [[
        {"x": 20, "y": 25, "width": 24, "height": 30, "score": 0.9},
        {"x": 105, "y": 35, "width": 48, "height": 54, "score": 0.9},
    ]]
    regions, crops = analyze_faces(image, None, None, boxes, "all", 16, 8, 0.25, 1.8, None, "balanced")

    encoded_shapes = []

    class IdentityVae:
        def spacial_compression_encode(self):
            return 8

        def encode(self, crop):
            encoded_shapes.append(tuple(crop.shape))
            return crop

        def decode(self, samples):
            return samples

    def identity_sampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
        return (latent,)

    monkeypatch.setattr(nodes, "common_ksampler", identity_sampler)
    processed, reports = process_face_crops(
        crops,
        regions,
        object(),
        IdentityVae(),
        object(),
        object(),
        "repair",
        0.42,
        12,
        5,
        1.0,
        "euler",
        "normal",
    )

    expected = [
        (1, ((region.target_size + 7) // 8) * 8, ((region.target_size + 7) // 8) * 8, 3)
        for region in regions.regions
    ]
    assert encoded_shapes == expected
    assert processed.shape[1:3] == (max(region.target_size for region in regions.regions),) * 2
    assert [report["processing_size"] for report in reports] == [region.target_size for region in regions.regions]
    assert [report["vae_aligned_size"] for report in reports] == [shape[1] for shape in expected]
    assert len(reports) == 2
