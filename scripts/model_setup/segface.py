from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .specs import SEGFACE_REQUIRED_TENSORS, SEGFACE_URL


def validate_segface(path: Path) -> None:
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            keys = set(weights.keys())
            if len(keys) != 463:
                raise RuntimeError(f"expected 463 tensors, found {len(keys)}")
            for name, expected_shape in SEGFACE_REQUIRED_TENSORS.items():
                if name not in keys:
                    raise RuntimeError(f"missing tensor {name}")
                actual_shape = tuple(weights.get_tensor(name).shape)
                if actual_shape != expected_shape:
                    raise RuntimeError(f"tensor {name} has shape {actual_shape}, expected {expected_shape}")
    except Exception as error:
        raise RuntimeError(f"Invalid converted SegFace weights at {path}: {error}") from error


def convert_segface(source: Path, destination: Path) -> None:
    if destination.is_file():
        validate_segface(destination)
        print(f"Already present and valid: {destination}")
        return
    print(f"Extracting SegFace inference weights to {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict_backbone", checkpoint)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("SegFace checkpoint has no state_dict_backbone")
    tensors = {str(key): value.detach().contiguous() for key, value in state.items() if torch.is_tensor(value)}
    if len(tensors) < 100:
        raise RuntimeError(f"Unexpected SegFace state dict: only {len(tensors)} tensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".safetensors", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(tensors, temporary, metadata={"format": "pt", "source": SEGFACE_URL})
        validate_segface(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
