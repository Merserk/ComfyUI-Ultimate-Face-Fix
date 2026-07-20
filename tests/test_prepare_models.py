from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_models.py"
SPEC = importlib.util.spec_from_file_location("ultimate_face_fix_prepare_models", SCRIPT)
assert SPEC and SPEC.loader
prepare_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_models)


def _valid_segface_tensors() -> dict[str, torch.Tensor]:
    tensors = {f"test_tensor_{index:03d}": torch.zeros(1) for index in range(460)}
    tensors.update(
        {
            "backbone.0.0.0.weight": torch.zeros(128, 3, 4, 4),
            "linear_fuse.weight": torch.zeros(256, 1024, 1, 1),
            "pe_layer.positional_encoding_gaussian_matrix": torch.zeros(2, 128),
        }
    )
    return tensors


def test_segface_validation_is_structural_not_serialization_hash(tmp_path):
    model = tmp_path / "segface.safetensors"
    save_file(_valid_segface_tensors(), model, metadata={"local_build": "different serializer metadata"})
    assert prepare_models.sha256(model) != "a2fb92765dbac1f81829738c8468e33819428b7319fe48e9508af7b963026e85"
    prepare_models.validate_segface(model)


def test_segface_validation_rejects_missing_weights(tmp_path):
    model = tmp_path / "invalid.safetensors"
    save_file({"only_one": torch.zeros(1)}, model)
    with pytest.raises(RuntimeError, match="expected 463 tensors"):
        prepare_models.validate_segface(model)


def test_python_download_fallback_writes_and_verifies_destination(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"face-fix-model-test")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "models" / "downloaded.bin"
    prepare_models.download(source.as_uri(), destination, expected)
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.with_suffix(".bin.part").exists()


def test_accelerated_hub_download_moves_file_without_leaving_cache(tmp_path, monkeypatch):
    def fake_hf_hub_download(*, filename, local_dir, **kwargs):
        downloaded = Path(local_dir) / filename
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"accelerated")
        (Path(local_dir) / ".cache").mkdir()
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_hf_hub_download))
    destination = tmp_path / "model.pt.part"
    assert prepare_models._hub_download("owner/repo", "folder/model.pt", "revision", destination)
    assert destination.read_bytes() == b"accelerated"
    assert not any(path.name.startswith(f".{destination.name}.download-") for path in tmp_path.iterdir())
