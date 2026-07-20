from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from model_setup.prepare import prepare_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Ultimate Face Fix model files for ComfyUI.")
    parser.add_argument("--comfy-root", type=Path, required=True, help="Directory containing folder_paths.py")
    parser.add_argument("--keep-segface-source", action="store_true")
    args = parser.parse_args()

    comfy_root = args.comfy_root.resolve(strict=True)
    prepare_models(comfy_root, keep_segface_source=args.keep_segface_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
