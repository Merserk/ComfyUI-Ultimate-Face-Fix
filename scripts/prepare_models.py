from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from model_setup.download import download, sha256
from model_setup.segface import convert_segface
from model_setup.specs import (
    MEDIAPIPE_SHA256,
    MEDIAPIPE_URL,
    SEGFACE_SOURCE_SHA256,
    SEGFACE_URL,
    YOLO_SHA256,
    YOLO_URL,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Ultimate Face Fix model files for ComfyUI.")
    parser.add_argument("--comfy-root", type=Path, required=True, help="Directory containing folder_paths.py")
    parser.add_argument("--keep-segface-source", action="store_true")
    args = parser.parse_args()

    comfy_root = args.comfy_root.resolve(strict=True)
    models = comfy_root / "models"
    detector_dir = models / "face_fix" / "detectors"
    parser_dir = models / "face_fix" / "parsers"
    legacy_cache_dir = parser_dir / ".setup_cache"
    legacy_segface_source = legacy_cache_dir / "segface_convnext_celeba_512_model_299.pt"
    segface_source = parser_dir / "segface_convnext_celeba_512_model_299.pt"
    segface_output = parser_dir / "segface_convnext_celeba_512.safetensors"
    yolo_output = detector_dir / "face_yolov9c.pt"
    mediapipe_output = models / "detection" / "mediapipe_face_fp32.safetensors"

    if not segface_output.is_file() and not segface_source.is_file() and legacy_segface_source.is_file():
        if sha256(legacy_segface_source) != SEGFACE_SOURCE_SHA256:
            raise RuntimeError(f"Invalid legacy SegFace download: {legacy_segface_source}")
        print(f"Reusing previous SegFace download: {legacy_segface_source}")
        os.replace(legacy_segface_source, segface_source)
    if not segface_output.is_file():
        download(
            SEGFACE_URL,
            segface_source,
            SEGFACE_SOURCE_SHA256,
            repo_id="kartiknarayan/SegFace",
            filename="convnext_celeba_512/model_299.pt",
            revision="5e093b03c0523f7f32a9845bbbc75ecb027c8bee",
        )
    convert_segface(segface_source, segface_output)
    download(
        YOLO_URL,
        yolo_output,
        YOLO_SHA256,
        repo_id="Bingsu/adetailer",
        filename="face_yolov9c.pt",
        revision="53cc19de382014514d9d4038601d261a7faa9b7b",
    )
    download(
        MEDIAPIPE_URL,
        mediapipe_output,
        MEDIAPIPE_SHA256,
        repo_id="Comfy-Org/mediapipe",
        filename="detection/mediapipe_face_fp32.safetensors",
        revision="b98d050e8bf406f14f063bdba697e5b5391bbbf5",
    )

    manifest = {
        "segface": {"path": str(segface_output), "sha256": sha256(segface_output)},
        "face_detector": {"path": str(yolo_output), "sha256": sha256(yolo_output)},
        "mediapipe": {"path": str(mediapipe_output), "sha256": sha256(mediapipe_output)},
    }
    if not args.keep_segface_source:
        segface_source.unlink(missing_ok=True)
    try:
        legacy_cache_dir.rmdir()
    except OSError:
        pass
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
