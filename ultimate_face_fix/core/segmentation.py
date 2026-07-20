from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .geometry import Box, box_area
from ..vendor import SegFaceCeleb


CELEBAMASK_CLASSES = (
    "background",
    "neck",
    "skin",
    "cloth",
    "l_ear",
    "r_ear",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "l_lip",
    "u_lip",
    "hair",
    "eye_g",
    "hat",
    "ear_r",
    "neck_l",
)

MASK_PRESETS = {
    "core_face": {"skin", "l_brow", "r_brow", "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip"},
    "portrait_safe": {
        "skin",
        "l_ear",
        "r_ear",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "nose",
        "mouth",
        "l_lip",
        "u_lip",
        "eye_g",
    },
    "head": {
        "skin",
        "l_ear",
        "r_ear",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "nose",
        "mouth",
        "l_lip",
        "u_lip",
        "hair",
        "eye_g",
        "hat",
        "ear_r",
    },
}


class SegFaceRunner:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SegFace parser was not found: {path}")
        self.model_path = str(path)
        self.model_name = path.name
        state = comfy.utils.load_torch_file(self.model_path, safe_load=True)
        if "state_dict_backbone" in state:
            state = state["state_dict_backbone"]
        state = {key.removeprefix("module."): value for key, value in state.items()}

        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.model = SegFaceCeleb().eval().to(offload_device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {len(missing)} keys, first: {missing[:3]}")
            if unexpected:
                details.append(f"unexpected {len(unexpected)} keys, first: {unexpected[:3]}")
            raise RuntimeError("SegFace checkpoint does not match ConvNeXt-512: " + "; ".join(details))
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=offload_device,
            size=comfy.model_management.module_size(self.model),
        )

    def parse(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"SegFace expects BHWC images, got {tuple(images.shape)}")
        comfy.model_management.load_model_gpu(self.patcher)
        inputs = F.interpolate(
            images[..., :3].movedim(-1, 1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(device=self.load_device, dtype=torch.float32)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.load_device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.load_device).view(1, 3, 1, 1)
        probabilities = self.model((inputs - mean) / std).softmax(dim=1)
        confidence, labels = probabilities.max(dim=1)
        intermediate = comfy.model_management.intermediate_device()
        return labels.to(intermediate), confidence.to(intermediate)


def semantic_mask(labels: torch.Tensor, preset: str, size: int) -> torch.Tensor:
    selected = MASK_PRESETS[preset]
    class_ids = [index for index, name in enumerate(CELEBAMASK_CLASSES) if name in selected]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        mask |= labels == class_id
    resized = F.interpolate(mask.float().unsqueeze(1), size=(size, size), mode="nearest")[:, 0]
    return resized


def semantic_mask_valid(mask: torch.Tensor, confidence: torch.Tensor, face_box: Box) -> bool:
    height, width = mask.shape[-2:]
    confidence = F.interpolate(confidence.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    area = float(mask.sum().item())
    expected = max(1.0, box_area(face_box))
    ratio = area / expected
    if area > 0:
        mean_confidence = float((confidence * mask).sum().item() / area)
    else:
        mean_confidence = 0.0
    center_x = int(round((face_box[0] + face_box[2]) * 0.5))
    center_y = int(round((face_box[1] + face_box[3]) * 0.5))
    center_x = max(0, min(width - 1, center_x))
    center_y = max(0, min(height - 1, center_y))
    center_hit = bool(mask[0, center_y, center_x] > 0.5)
    return 0.15 <= ratio <= 1.80 and mean_confidence >= 0.35 and (center_hit or ratio >= 0.35)


def sam_box_mask(image: torch.Tensor, face_box: Box, sam_model, refine_iterations: int = 2) -> torch.Tensor | None:
    if sam_model is None:
        return None
    comfy.model_management.load_model_gpu(sam_model)
    diffusion_model = getattr(getattr(sam_model, "model", None), "diffusion_model", None)
    if diffusion_model is None or not hasattr(diffusion_model, "forward_segment"):
        raise ValueError("The optional SAM input must be a SAM 3 or SAM 3.1 MODEL")
    device = comfy.model_management.get_torch_device()
    dtype = sam_model.model.get_dtype()
    size = image.shape[0]
    frame = F.interpolate(
        image[..., :3].movedim(-1, 0).unsqueeze(0),
        size=(1008, 1008),
        mode="bilinear",
        align_corners=False,
    ).to(device=device, dtype=dtype)
    scale = 1008.0 / size
    x1, y1, x2, y2 = face_box
    box_input = torch.tensor(
        [[[(x1 * scale), (y1 * scale)], [(x2 * scale), (y2 * scale)]]],
        device=device,
        dtype=dtype,
    )
    logits = diffusion_model.forward_segment(frame, box_inputs=box_input)
    for _ in range(max(0, refine_iterations - 1)):
        logits = diffusion_model.forward_segment(frame, mask_inputs=logits)
    mask = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)[0]
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).float().to(comfy.model_management.intermediate_device())
