from __future__ import annotations

import importlib

import torch

from .geometry import reflection_indices, stack_square_crops
from .types import FaceFixRegions


REPAIR_DENOISE = {
    "detail": 0.28,
    "repair": 0.42,
    "reconstruct": 0.58,
}


def normalize_vae_images(decoded: torch.Tensor) -> torch.Tensor:
    """Normalize ComfyUI image and single-frame video VAE output to one HWC image."""
    if not isinstance(decoded, torch.Tensor):
        raise ValueError(f"The connected VAE returned {type(decoded).__name__}, expected a torch.Tensor")
    if decoded.ndim == 5:
        decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
    if decoded.ndim != 4:
        raise ValueError(
            "The connected VAE must decode to BHWC or BTHWC image data; "
            f"received shape {tuple(decoded.shape)}"
        )
    if decoded.shape[0] != 1:
        raise ValueError(
            "Each face crop must decode to exactly one image; "
            f"received {decoded.shape[0]} images with shape {tuple(decoded.shape)}"
        )
    if decoded.shape[-1] not in (3, 4):
        raise ValueError(
            "The connected VAE must return channel-last RGB or RGBA images; "
            f"received shape {tuple(decoded.shape)}"
        )
    return decoded[0, ..., :3].clamp(0.0, 1.0)


def _vae_alignment(vae) -> int:
    get_alignment = getattr(vae, "spacial_compression_encode", None)
    if get_alignment is None:
        return 1
    try:
        return max(1, int(get_alignment()))
    except (TypeError, ValueError):
        return 1


def _pad_native_crop(crop: torch.Tensor, alignment: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    side = crop.shape[1]
    aligned_side = ((side + alignment - 1) // alignment) * alignment
    padding = aligned_side - side
    before = padding // 2
    after = padding - before
    if padding == 0:
        return crop, (0, 0, 0, 0)
    y_indices = reflection_indices(-before, aligned_side, side, crop.device)
    x_indices = reflection_indices(-before, aligned_side, side, crop.device)
    return crop.index_select(1, y_indices).index_select(2, x_indices), (before, before, after, after)


def process_face_crops(
    crops: torch.Tensor,
    regions: FaceFixRegions,
    model,
    vae,
    positive,
    negative,
    repair_mode: str,
    custom_denoise: float,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
) -> torch.Tensor:
    if regions.face_count == 0:
        return crops.clone()
    denoise = custom_denoise if repair_mode == "custom" else REPAIR_DENOISE[repair_mode]
    comfy_nodes = importlib.import_module("nodes")
    outputs = []
    for crop_index, region in enumerate(regions.regions):
        crop = crops[crop_index : crop_index + 1, : region.target_size, : region.target_size, :3]
        vae_padding = (0, 0, 0, 0)
        sample_crop = crop
        if regions.target_size is None:
            sample_crop, vae_padding = _pad_native_crop(crop, _vae_alignment(vae))
        latent = {"samples": vae.encode(sample_crop)}
        face_seed = (int(seed) + region.seed_offset) & 0xFFFFFFFFFFFFFFFF
        sampled = comfy_nodes.common_ksampler(
            model,
            face_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]
        decoded = normalize_vae_images(vae.decode(sampled["samples"]))
        expected_side = sample_crop.shape[1]
        if decoded.shape[:2] != (expected_side, expected_side):
            raise ValueError(
                "The connected VAE changed the face crop dimensions: "
                f"expected {expected_side}x{expected_side}, got {decoded.shape[1]}x{decoded.shape[0]} "
                f"after output normalization"
            )
        left, top, _, _ = vae_padding
        decoded = decoded[top : top + region.target_size, left : left + region.target_size]
        outputs.append(decoded)
    return stack_square_crops(outputs)
