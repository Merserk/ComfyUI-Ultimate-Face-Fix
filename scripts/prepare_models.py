from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

import torch
from safetensors.torch import save_file


SEGFACE_URL = (
    "https://huggingface.co/kartiknarayan/SegFace/resolve/"
    "5e093b03c0523f7f32a9845bbbc75ecb027c8bee/convnext_celeba_512/model_299.pt"
)
YOLO_URL = (
    "https://huggingface.co/Bingsu/adetailer/resolve/"
    "53cc19de382014514d9d4038601d261a7faa9b7b/face_yolov9c.pt"
)
MEDIAPIPE_URL = (
    "https://huggingface.co/Comfy-Org/mediapipe/resolve/"
    "b98d050e8bf406f14f063bdba697e5b5391bbbf5/detection/"
    "mediapipe_face_fp32.safetensors"
)
SEGFACE_SOURCE_SHA256 = "3cd535bfcfab4e5c67b7df7015370c2b18cb9a4044ecb00c7c1fa8555558c507"
SEGFACE_RUNTIME_SHA256 = "a2fb92765dbac1f81829738c8468e33819428b7319fe48e9508af7b963026e85"
YOLO_SHA256 = "d02fe493c31e1bbc6450f4dc6f1db86a02a59322ff1f6d318da0661d72ddd084"
MEDIAPIPE_SHA256 = "a98c4806081d40eba35102a0f6dc0000c2e1388b72cf24e691703d0605bd888a"


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    if destination.is_file():
        actual = sha256(destination)
        if expected_sha256 is None or actual == expected_sha256:
            print(f"Already present: {destination}")
            return
        raise RuntimeError(f"Refusing to replace {destination.name}; SHA-256 is {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}\n       to {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Ultimate-Face-Fix/setup"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if expected_sha256 is not None:
        actual = sha256(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {destination.name}: {actual}")
    os.replace(partial, destination)


def convert_segface(source: Path, destination: Path) -> None:
    if destination.is_file():
        actual = sha256(destination)
        if actual == SEGFACE_RUNTIME_SHA256:
            print(f"Already present: {destination}")
            return
        raise RuntimeError(f"Refusing to replace {destination.name}; SHA-256 is {actual}")
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
        actual = sha256(temporary)
        if actual != SEGFACE_RUNTIME_SHA256:
            raise RuntimeError(f"Converted SegFace SHA-256 mismatch: {actual}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Ultimate Face Fix model files for ComfyUI.")
    parser.add_argument("--comfy-root", type=Path, required=True, help="Directory containing folder_paths.py")
    parser.add_argument("--keep-segface-source", action="store_true")
    args = parser.parse_args()

    comfy_root = args.comfy_root.resolve(strict=True)
    models = comfy_root / "models"
    detector_dir = models / "face_fix" / "detectors"
    parser_dir = models / "face_fix" / "parsers"
    cache_dir = parser_dir / ".setup_cache"
    segface_source = cache_dir / "segface_convnext_celeba_512_model_299.pt"
    segface_output = parser_dir / "segface_convnext_celeba_512.safetensors"
    yolo_output = detector_dir / "face_yolov9c.pt"
    mediapipe_output = models / "detection" / "mediapipe_face_fp32.safetensors"

    if not segface_output.is_file():
        download(SEGFACE_URL, segface_source, SEGFACE_SOURCE_SHA256)
    convert_segface(segface_source, segface_output)
    download(YOLO_URL, yolo_output, YOLO_SHA256)
    download(MEDIAPIPE_URL, mediapipe_output, MEDIAPIPE_SHA256)

    manifest = {
        "segface": {"path": str(segface_output), "sha256": sha256(segface_output)},
        "face_detector": {"path": str(yolo_output), "sha256": sha256(yolo_output)},
        "mediapipe": {"path": str(mediapipe_output), "sha256": sha256(mediapipe_output)},
    }
    manifest_path = Path(__file__).resolve().parents[1] / "MODEL_MANIFEST.generated.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.keep_segface_source:
        segface_source.unlink(missing_ok=True)
        try:
            cache_dir.rmdir()
        except OSError:
            pass
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
