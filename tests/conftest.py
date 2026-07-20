from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
installed_root = PACKAGE_ROOT.parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", installed_root)).resolve()
if not (COMFY_ROOT / "folder_paths.py").is_file():
    raise RuntimeError("Set COMFYUI_ROOT to the directory containing folder_paths.py before running tests")
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

if "ultimate_face_fix" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "ultimate_face_fix",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
