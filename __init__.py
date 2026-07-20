from __future__ import annotations

import os
import sys

from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension


# Pytest imports a custom-node directory named with hyphens as a top-level
# ``__init__`` module. Give that diagnostic import a valid package alias so
# relative imports behave exactly as they do under ComfyUI's loader.
if not __package__:
    __package__ = "ultimate_face_fix"
    __path__ = [os.path.dirname(__file__)]
    sys.modules.setdefault(__package__, sys.modules[__name__])


MODEL_ROOT = os.path.join(folder_paths.models_dir, "face_fix")
DETECTOR_ROOT = os.path.join(MODEL_ROOT, "detectors")
PARSER_ROOT = os.path.join(MODEL_ROOT, "parsers")
os.makedirs(DETECTOR_ROOT, exist_ok=True)
os.makedirs(PARSER_ROOT, exist_ok=True)
folder_paths.folder_names_and_paths["face_fix_detectors"] = ([DETECTOR_ROOT], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["face_fix_parsers"] = ([PARSER_ROOT], folder_paths.supported_pt_extensions)

from .nodes import FaceFixAnalyze, FaceFixComposite, LoadFaceFixDetector, LoadFaceFixParser, UltimateFaceFix


class UltimateFaceFixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            LoadFaceFixDetector,
            LoadFaceFixParser,
            UltimateFaceFix,
            FaceFixAnalyze,
            FaceFixComposite,
        ]


async def comfy_entrypoint() -> UltimateFaceFixExtension:
    return UltimateFaceFixExtension()
