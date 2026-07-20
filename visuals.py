from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from .regions import FaceFixRegions


def detection_preview(images: torch.Tensor, regions: FaceFixRegions) -> torch.Tensor:
    result = []
    colors = ((0, 255, 80), (0, 180, 255), (255, 170, 0), (255, 70, 140))
    for source_index, image in enumerate(images):
        array = image[..., :3].mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        canvas = Image.fromarray(array, "RGB")
        draw = ImageDraw.Draw(canvas)
        width = max(2, int(round(min(canvas.size) / 256)))
        for region in regions.regions:
            if region.source_index != source_index:
                continue
            color = colors[region.face_index % len(colors)]
            x1, y1, x2, y2 = region.detection_box
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            draw.text((x1 + width, max(0, y1 - 14)), f"{region.face_index + 1} {region.confidence:.2f}", fill=color)
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).div_(255.0)
        result.append(tensor.to(device=images.device, dtype=images.dtype))
    return torch.stack(result)


def mask_preview(images: torch.Tensor, masks: torch.Tensor, regions: FaceFixRegions) -> torch.Tensor:
    overlay_color = torch.tensor((0.08, 1.0, 0.35), dtype=images.dtype, device=images.device)
    alpha = masks.to(device=images.device, dtype=images.dtype).unsqueeze(-1).mul(0.45)
    overlay = images[..., :3] * (1.0 - alpha) + overlay_color * alpha
    return detection_preview(overlay.clamp(0.0, 1.0), regions)
