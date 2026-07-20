from __future__ import annotations

import os
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension


MODEL_ROOT = os.path.join(folder_paths.models_dir, "face_fix")
DETECTOR_ROOT = os.path.join(MODEL_ROOT, "detectors")
PARSER_ROOT = os.path.join(MODEL_ROOT, "parsers")
os.makedirs(DETECTOR_ROOT, exist_ok=True)
os.makedirs(PARSER_ROOT, exist_ok=True)
folder_paths.folder_names_and_paths["face_fix_detectors"] = ([DETECTOR_ROOT], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths["face_fix_parsers"] = ([PARSER_ROOT], folder_paths.supported_pt_extensions)

from .ultimate_face_fix import (
    LoadFaceFixDetector,
    LoadFaceFixParser,
    UltimateFaceFix,
    UltimateFaceFixExtract,
    UltimateFaceFixProcess,
)


class UltimateFaceFixExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            LoadFaceFixDetector,
            LoadFaceFixParser,
            UltimateFaceFixExtract,
            UltimateFaceFixProcess,
            UltimateFaceFix,
        ]


async def comfy_entrypoint() -> UltimateFaceFixExtension:
    return UltimateFaceFixExtension()
