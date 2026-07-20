from __future__ import annotations

import os
from pathlib import Path

from .prepare import prepare_models


def find_comfy_root() -> Path:
    candidates = []
    configured = os.environ.get("COMFYUI_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())

    repository_root = Path(__file__).resolve().parents[2]
    for start in (repository_root, Path.cwd().resolve()):
        candidates.extend((start, *start.parents))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "folder_paths.py").is_file():
            return resolved
    raise RuntimeError(
        "Could not locate the ComfyUI root. Set COMFYUI_PATH or run the manual "
        "scripts/prepare_models.py command with --comfy-root."
    )


def run_manager_hook() -> int:
    comfy_root = find_comfy_root()
    print(f"Ultimate Face Fix: preparing required models in {comfy_root / 'models'}")
    prepare_models(comfy_root)
    print("Ultimate Face Fix: all required models are ready.")
    return 0
